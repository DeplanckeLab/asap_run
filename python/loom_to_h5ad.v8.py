#!/usr/bin/env python3
"""Convert a Loom file to H5AD (AnnData).

Uses anndata + loompy. Applies NumPy 2.x compatibility aliases that older
loompy still references (np.string_, np.unicode_).

Also recovers loom row/col attrs that loompy materializes as None (non-ASCII
UTF-8 such as U+00A0) by re-reading the same HDF5 datasets with UTF-8 decode.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys


def write_output_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def fail(message: str, output_json: str | None = None) -> None:
    payload = {"displayed_error": message}
    if output_json:
        write_output_json(output_json, payload)
    else:
        print(json.dumps(payload), file=sys.stderr)
    sys.exit(1)


def apply_numpy_compat() -> None:
    import numpy as np

    if not hasattr(np, "string_"):
        np.string_ = np.bytes_
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_


def decode_loom_attr_values(raw):
    """Decode a loom HDF5 attr array to unicode object strings (UTF-8 safe)."""
    import numpy as np

    a = np.asanyarray(raw)
    scalar = a.ndim == 0
    if scalar:
        a = np.reshape(a, (1,))

    if a.size and (
        np.issubdtype(a.dtype, np.number) or np.issubdtype(a.dtype, np.bool_)
    ):
        return raw

    out = []
    for x in a.ravel():
        if x is None:
            out.append("")
        elif isinstance(x, (bytes, bytearray, np.bytes_)):
            out.append(html.unescape(bytes(x).decode("utf-8")))
        else:
            out.append(html.unescape(str(x)))
    result = np.array(out, dtype=object).reshape(a.shape)
    return result[0] if scalar else result


def install_loompy_none_attr_recovery() -> None:
    """When loompy returns None for an attr, re-read it from HDF5 with UTF-8.

    loompy.materialize_attr_values can leave attrs as None when values contain
    non-ASCII UTF-8 (e.g. NBSP in tissue_source). anndata.read_loom then fails
    on ``v.ndim``. The data is still present in the loom file.
    """
    from loompy.attribute_manager import AttributeManager

    if getattr(AttributeManager.__getattr__, "_asap_none_attr_recovery", False):
        return

    _orig_getattr = AttributeManager.__getattr__

    def __getattr__(self, name: str):
        vals = _orig_getattr(self, name)
        if vals is not None:
            return vals
        ds = self.__dict__.get("ds")
        if ds is None:
            return vals
        axis_prefix = ["/row_attrs/", "/col_attrs/"][self.axis]
        raw = ds._file[axis_prefix][name][:]
        recovered = decode_loom_attr_values(raw)
        self.__dict__["storage"][name] = recovered
        print(
            f"Recovered loom attr {axis_prefix}{name} via UTF-8 HDF5 re-read",
            flush=True,
        )
        return recovered

    __getattr__._asap_none_attr_recovery = True  # type: ignore[attr-defined]
    AttributeManager.__getattr__ = __getattr__  # type: ignore[method-assign]


def drop_redundant_layer_x(adata) -> bool:
    """Remove adata.layers['X'] when it is identical to adata.X.

    ASAP looms often carry both /matrix and /layers/X with the same values.
    read_loom maps both into X and layers['X'], doubling h5ad size. scFAIR
    requires X, not a duplicate layers['X'].
    """
    from scipy import sparse

    if "X" not in adata.layers:
        return False

    x = adata.X
    layer = adata.layers["X"]
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    else:
        x = x.tocsr()
    if not sparse.issparse(layer):
        layer = sparse.csr_matrix(layer)
    else:
        layer = layer.tocsr()

    if x.shape != layer.shape or (x != layer).nnz != 0:
        return False

    del adata.layers["X"]
    print("Dropped redundant layers['X'] (identical to X)", flush=True)
    return True


def _decode_loom_attr_scalar(raw):
    """Normalize a loom /attrs value to a Python scalar suitable for adata.uns."""
    import numpy as np

    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        return bytes(raw).decode("utf-8")
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            return _decode_loom_attr_scalar(raw.item())
        if raw.ndim == 1 and raw.size == 1:
            return _decode_loom_attr_scalar(raw[0])
        return [_decode_loom_attr_scalar(v) for v in raw.tolist()]
    if isinstance(raw, np.generic):
        return raw.item()
    return raw


def copy_loom_attrs_to_uns(loom_file: str, adata) -> int:
    """Copy loom /attrs/* into adata.uns (read_loom leaves uns empty).

    ASAP scFAIR metadata (title, organism*, schema_*, ensembl_*) lives in loom
    global attrs and must be present under uns for h5ad compliance.
    """
    import h5py

    n = 0
    with h5py.File(loom_file, "r") as hf:
        if "attrs" not in hf:
            return 0
        for key in hf["attrs"].keys():
            raw = hf["attrs"][key][()]
            adata.uns[key] = _decode_loom_attr_scalar(raw)
            n += 1
    if n:
        print(f"Copied {n} loom /attrs keys into uns", flush=True)
    return n


def _parse_dburl(dburl: str) -> tuple[str, int, str]:
    from urllib.parse import urlparse

    u = urlparse("http://" + dburl) if "://" not in dburl else urlparse(dburl)
    host = u.hostname
    port = u.port or 5432
    dbname = (u.path or "").lstrip("/")
    if not host or not dbname:
        raise ValueError(f"Invalid --dburl {dburl!r}; expected HOST:PORT/DBNAME")
    return host, int(port), dbname


def _tax_id_from_ontology_term(term: str) -> int | None:
    s = str(term or "").strip()
    if s.upper().startswith("NCBITAXON:"):
        tail = s.split(":", 1)[1]
        if tail.isdigit():
            return int(tail)
    if s.isdigit():
        return int(s)
    return None


def align_feature_name_to_gene_db(adata, *, dburl: str) -> None:
    """Set var['feature_name'] from ASAP genes.name by Accession (scFAIR).

    Var index must be Ensembl Accession. Spike-ins use
    '{ERCC-id} (spike-in control)'. Gene symbols copied from CellXGENE are not
    always the genes.name value scFAIR expects.
    """
    import os
    import re

    import numpy as np
    import psycopg2
    from psycopg2.extras import RealDictCursor

    if adata.n_vars <= 0:
        return

    tax_id = _tax_id_from_ontology_term(adata.uns.get("organism_ontology_term_id", ""))
    if tax_id is None:
        raise ValueError(
            "uns/organism_ontology_term_id is required to align feature_name "
            "(NCBITaxon:<tax_id>)"
        )

    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    if not user or not password:
        raise ValueError(
            "POSTGRES_USER and POSTGRES_PASSWORD are required to align feature_name"
        )

    host, port, dbname = _parse_dburl(dburl)
    accessions = [str(x) for x in adata.var_names.tolist()]
    accessions_nover = [re.sub(r"\.\d+$", "", a) for a in accessions]

    sql = """
        SELECT g.ensembl_id, g.name
        FROM genes g
        JOIN organisms o ON o.id = g.organism_id
        WHERE o.tax_id = %s
          AND regexp_replace(g.ensembl_id, '\\.\\d+$', '') = ANY(%s)
    """
    with psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (tax_id, accessions_nover))
            rows = cur.fetchall()

    name_by_id: dict[str, str] = {}
    for row in rows:
        eid = re.sub(r"\.\d+$", "", str(row["ensembl_id"] or "").strip())
        name = str(row["name"] or "").strip()
        if eid:
            name_by_id[eid] = name or eid

    biotypes = (
        adata.var["feature_biotype"].astype(str).tolist()
        if "feature_biotype" in adata.var.columns
        else ["gene"] * adata.n_vars
    )

    out: list[str] = []
    missing = 0
    for acc, bt in zip(accessions, biotypes):
        if bt == "spike-in" or acc.upper().startswith("ERCC"):
            ercc = acc if acc.upper().startswith("ERCC") else acc
            out.append(f"{ercc} (spike-in control)")
            continue
        key = re.sub(r"\.\d+$", "", acc)
        name = name_by_id.get(key)
        if not name:
            missing += 1
            name = acc
        out.append(name)

    adata.var["feature_name"] = np.array(out, dtype=object)
    print(
        f"Aligned feature_name from gene DB (tax_id={tax_id}, "
        f"looked_up={len(name_by_id)}, missing_fallback_to_index={missing})",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Loom to H5AD")
    parser.add_argument("-i", "--input", required=True, help="Absolute path to input .loom")
    parser.add_argument("-o", "--output", required=True, help="Absolute path to output .h5ad")
    parser.add_argument("-d", "--output_dir", required=True, help="Run directory for output.json")
    parser.add_argument(
        "--dburl",
        required=True,
        help="ASAP gene DB HOST:PORT/DBNAME (same as parse.v8.py --dburl)",
    )
    args = parser.parse_args(argv)

    loom_file = args.input
    h5ad_file = args.output
    output_dir = args.output_dir
    output_json = os.path.join(output_dir, "output.json")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(h5ad_file) or ".", exist_ok=True)

    if not os.path.isfile(loom_file):
        fail(f"Input loom not found: {loom_file}", output_json)

    try:
        apply_numpy_compat()
        import loompy  # noqa: F401 - load before patching AttributeManager
        import anndata as ad
        from anndata.io import read_loom

        install_loompy_none_attr_recovery()
        ad.settings.allow_write_nullable_strings = True
        print(f"Converting {loom_file} -> {h5ad_file}", flush=True)
        # scFAIR: obs index = CellID, var index = Accession (Ensembl), not Gene symbols.
        adata = read_loom(loom_file, obs_names="CellID", var_names="Accession")
        drop_redundant_layer_x(adata)
        copy_loom_attrs_to_uns(loom_file, adata)
        align_feature_name_to_gene_db(adata, dburl=args.dburl)
        adata.write_h5ad(h5ad_file)
    except Exception as exc:  # noqa: BLE001 - surface any convert failure in output.json
        fail(f"Loom to H5AD conversion failed: {exc}", output_json)

    if not os.path.isfile(h5ad_file) or os.path.getsize(h5ad_file) <= 0:
        fail("H5AD output was not written or is empty", output_json)

    payload = {
        "status": "success",
        "input_loom": loom_file,
        "output_h5ad": h5ad_file,
        "output_h5ad_bytes": os.path.getsize(h5ad_file),
        "n_obs": int(getattr(adata, "n_obs", 0)),
        "n_vars": int(getattr(adata, "n_vars", 0)),
    }
    write_output_json(output_json, payload)
    print(
        f"Wrote {h5ad_file} ({payload['output_h5ad_bytes']} bytes, "
        f"{payload['n_obs']} x {payload['n_vars']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

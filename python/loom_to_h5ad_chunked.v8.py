#!/usr/bin/env python3
"""Chunked Loom -> H5AD converter with bounded RAM (AnnData-native encoding).

Does not replace loom_to_h5ad.v8.py. Streams cell chunks into on-disk CSR
scratch, then builds one AnnData and write_h5ad. Peak RAM is about one full
sparse X (plus layer X if present) at assemble/write time — not N chunk
matrices plus a vstack copy.

Mirrors anndata.io.read_loom conventions used by ASAP looms:
  - loom matrix is genes x cells; AnnData X is cells x genes (float32 CSR)
  - CellID -> obs index; Accession -> var index (scFAIR); Gene kept as var column
  - 1D col/row attrs -> obs/var columns; 2D col attrs -> obsm
  - named loom layer "X" -> adata.layers["X"] when present and not identical to /matrix

CLI is compatible with loom_to_h5ad.v8.py (-i/-o/-d) plus --chunk-cells.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
import shutil
import sys
import tempfile


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
    """When loompy returns None for an attr, re-read it from HDF5 with UTF-8."""
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
    """Copy loom /attrs/* into adata.uns (scFAIR uns metadata)."""
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
    """Set var['feature_name'] from ASAP genes.name by Accession (scFAIR)."""
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
            out.append(f"{acc} (spike-in control)")
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


def _as_1d_object_strings(values) -> list[str]:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D names, got shape {arr.shape}")
    out: list[str] = []
    for v in arr:
        if isinstance(v, (bytes, bytearray)):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def extract_obs_var_obsm(ds):
    """Build obs / var / obsm like anndata.io.read_loom for ASAP looms."""
    import numpy as np
    import pandas as pd

    n_genes, n_cells = ds.shape

    obs_data: dict = {}
    obsm: dict = {}
    for key in ds.ca.keys():
        vals = np.asarray(ds.ca[key])
        if vals.ndim == 2:
            obsm[key] = vals
        elif vals.ndim == 1:
            if key in {"CellID", "obs_names", "cell_names", "CellName"}:
                continue
            obs_data[key] = vals
        else:
            raise ValueError(f"Unsupported ca[{key!r}] ndim={vals.ndim}")

    if "CellID" in ds.ca:
        obs_index = _as_1d_object_strings(ds.ca["CellID"])
    else:
        raise ValueError("Loom is missing col_attrs/CellID (required for obs index)")
    if len(obs_index) != n_cells:
        raise ValueError(f"CellID length {len(obs_index)} != n_cells {n_cells}")
    obs = pd.DataFrame(obs_data, index=pd.Index(obs_index, name="CellID"))

    var_data: dict = {}
    for key in ds.ra.keys():
        vals = np.asarray(ds.ra[key])
        if vals.ndim != 1:
            raise ValueError(f"Unsupported ra[{key!r}] ndim={vals.ndim}")
        if key in {"Accession", "var_names", "gene_ids"}:
            continue
        var_data[key] = vals

    if "Accession" not in ds.ra:
        raise ValueError("Loom is missing row_attrs/Accession (required for var index / scFAIR)")
    var_index = _as_1d_object_strings(ds.ra["Accession"])
    if len(var_index) != n_genes:
        raise ValueError(f"Accession length {len(var_index)} != n_genes {n_genes}")
    if len(set(var_index)) != n_genes:
        raise ValueError(
            f"Accession values are not unique ({len(set(var_index))} unique / {n_genes})"
        )
    var = pd.DataFrame(var_data, index=pd.Index(var_index, name="Accession"))

    return obs, var, obsm


def read_matrix_chunk_csr(dataset, start: int, end: int):
    """Read loom genes x cells slice and return cells x genes float32 CSR."""
    import numpy as np
    from scipy import sparse

    block = dataset[:, start:end]
    if hasattr(block, "toarray"):
        block = block.toarray()
    block = np.asarray(block, dtype=np.float32, order="C")
    return sparse.csr_matrix(block.T)


def _init_csr_scratch(hf, name: str, n_rows: int):
    """Growing on-disk CSR buffers (avoids holding all chunk matrices in RAM)."""
    g = hf.create_group(name)
    g.create_dataset("data", shape=(0,), maxshape=(None,), dtype="float32", chunks=True)
    g.create_dataset("indices", shape=(0,), maxshape=(None,), dtype="int32", chunks=True)
    g.create_dataset(
        "indptr",
        shape=(n_rows + 1,),
        dtype="int64",
        fillvalue=0,
    )
    g.attrs["nnz"] = 0
    g.attrs["n_rows_filled"] = 0
    return g


def _append_csr_chunk(group, X) -> None:
    """Append one CSR block (rows) onto scratch datasets."""
    import numpy as np

    X = X.tocsr()
    data = np.asarray(X.data, dtype=np.float32)
    indices = np.asarray(X.indices, dtype=np.int32)
    indptr = np.asarray(X.indptr, dtype=np.int64)

    nnz0 = int(group.attrs["nnz"])
    n_rows0 = int(group.attrs["n_rows_filled"])
    n_new = int(X.shape[0])
    if n_rows0 + n_new > group["indptr"].shape[0] - 1:
        raise ValueError(
            f"CSR scratch overflow: filled={n_rows0} + {n_new} "
            f"> capacity={group['indptr'].shape[0] - 1}"
        )

    n_data = data.shape[0]
    group["data"].resize((nnz0 + n_data,))
    group["indices"].resize((nnz0 + n_data,))
    group["data"][nnz0 : nnz0 + n_data] = data
    group["indices"][nnz0 : nnz0 + n_data] = indices
    # Skip indptr[0] (==0); offset remaining entries by existing nnz.
    group["indptr"][n_rows0 + 1 : n_rows0 + n_new + 1] = indptr[1:] + nnz0
    group.attrs["nnz"] = nnz0 + n_data
    group.attrs["n_rows_filled"] = n_rows0 + n_new


def _csr_from_scratch(group, n_rows: int, n_cols: int):
    """Load one CSR from scratch (single final copy, not per-chunk list + vstack)."""
    import numpy as np
    from scipy import sparse

    if int(group.attrs["n_rows_filled"]) != n_rows:
        raise ValueError(
            f"CSR row mismatch: filled={group.attrs['n_rows_filled']} expected={n_rows}"
        )
    data = np.asarray(group["data"][:], dtype=np.float32)
    indices = np.asarray(group["indices"][:], dtype=np.int32)
    indptr = np.asarray(group["indptr"][: n_rows + 1], dtype=np.int64)
    return sparse.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))


def convert_chunked(
    loom_file: str,
    h5ad_file: str,
    *,
    chunk_cells: int,
    tmp_parent: str,
    dburl: str,
) -> tuple[int, int]:
    import anndata as ad
    import h5py
    import loompy
    import numpy as np

    ad.settings.allow_write_nullable_strings = True
    install_loompy_none_attr_recovery()

    with loompy.connect(loom_file, mode="r") as ds:
        n_genes, n_cells = (int(ds.shape[0]), int(ds.shape[1]))
        if n_genes <= 0 or n_cells <= 0:
            raise ValueError(f"Invalid loom shape {ds.shape}")
        obs, var, obsm = extract_obs_var_obsm(ds)

    if chunk_cells <= 0:
        raise ValueError("chunk_cells must be positive")

    tmp_dir = tempfile.mkdtemp(prefix="loom2h5ad_chunks_", dir=tmp_parent)
    scratch_path = os.path.join(tmp_dir, "csr_scratch.h5")
    try:
        with h5py.File(scratch_path, "w") as scratch:
            x_scratch = _init_csr_scratch(scratch, "X", n_cells)
            layer_scratch = None

            with h5py.File(loom_file, "r") as hf:
                if "matrix" not in hf:
                    raise ValueError("Loom file has no /matrix dataset")
                matrix_ds = hf["matrix"]
                layer_x_ds = (
                    hf["layers"]["X"] if "layers" in hf and "X" in hf["layers"] else None
                )
                layer_differs = False

                n_chunks = (n_cells + chunk_cells - 1) // chunk_cells
                for chunk_i, start in enumerate(range(0, n_cells, chunk_cells)):
                    end = min(start + chunk_cells, n_cells)
                    print(
                        f"Chunk {chunk_i + 1}/{n_chunks}: cells [{start}:{end}) "
                        f"({end - start} cells)",
                        flush=True,
                    )
                    X = read_matrix_chunk_csr(matrix_ds, start, end)
                    _append_csr_chunk(x_scratch, X)
                    if layer_x_ds is not None and not layer_differs:
                        L = read_matrix_chunk_csr(layer_x_ds, start, end)
                        if (X != L).nnz != 0:
                            layer_differs = True
                            print(
                                "loom layer 'X' differs from /matrix; "
                                "will keep layers['X'] in h5ad",
                                flush=True,
                            )
                        del L
                    del X
                    gc.collect()

                if layer_x_ds is not None and layer_differs:
                    layer_scratch = _init_csr_scratch(scratch, "layer_X", n_cells)
                    print(
                        f"Second pass: writing non-redundant loom layer 'X' "
                        f"({n_chunks} chunks)",
                        flush=True,
                    )
                    for chunk_i, start in enumerate(range(0, n_cells, chunk_cells)):
                        end = min(start + chunk_cells, n_cells)
                        L = read_matrix_chunk_csr(layer_x_ds, start, end)
                        _append_csr_chunk(layer_scratch, L)
                        del L
                        gc.collect()
                elif layer_x_ds is not None:
                    print(
                        "Skipping redundant loom layer 'X' (identical to /matrix)",
                        flush=True,
                    )

            print(f"Assembling CSR from scratch -> {h5ad_file}", flush=True)
            if os.path.exists(h5ad_file):
                os.remove(h5ad_file)
            # Reuse obs/var/obsm already in memory (no chunk reload / vstack of parts).
            # Peak RAM is about one full sparse X (+ layer only if it differs).
            X = _csr_from_scratch(x_scratch, n_cells, n_genes)
            merged = ad.AnnData(X=X, obs=obs, var=var)
            del X
            for key, arr in obsm.items():
                merged.obsm[key] = np.asarray(arr)
            if layer_scratch is not None:
                merged.layers["X"] = _csr_from_scratch(layer_scratch, n_cells, n_genes)
            copy_loom_attrs_to_uns(loom_file, merged)
            align_feature_name_to_gene_db(merged, dburl=dburl)
            merged.write_h5ad(h5ad_file)
            del merged
            gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return n_cells, n_genes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Loom to H5AD with chunked, AnnData-native writes"
    )
    parser.add_argument("-i", "--input", required=True, help="Absolute path to input .loom")
    parser.add_argument("-o", "--output", required=True, help="Absolute path to output .h5ad")
    parser.add_argument("-d", "--output_dir", required=True, help="Run directory for output.json")
    parser.add_argument(
        "--chunk-cells",
        type=int,
        default=int(os.environ.get("LOOM_TO_H5AD_CHUNK_CELLS", "2000")),
        help="Cells per chunk (default 2000, or LOOM_TO_H5AD_CHUNK_CELLS)",
    )
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
        print(
            f"Converting {loom_file} -> {h5ad_file} "
            f"(chunk_cells={args.chunk_cells}, converter=chunked)",
            flush=True,
        )
        n_obs, n_vars = convert_chunked(
            loom_file,
            h5ad_file,
            chunk_cells=args.chunk_cells,
            tmp_parent=output_dir,
            dburl=args.dburl,
        )
    except Exception as exc:  # noqa: BLE001 - surface any convert failure in output.json
        fail(f"Loom to H5AD conversion failed: {exc}", output_json)

    if not os.path.isfile(h5ad_file) or os.path.getsize(h5ad_file) <= 0:
        fail("H5AD output was not written or is empty", output_json)

    payload = {
        "status": "success",
        "input_loom": loom_file,
        "output_h5ad": h5ad_file,
        "output_h5ad_bytes": os.path.getsize(h5ad_file),
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
        "chunk_cells": int(args.chunk_cells),
        "converter": "chunked_anndata_concat",
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

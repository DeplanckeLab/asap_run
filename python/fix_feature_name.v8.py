#!/usr/bin/env python3
"""Rewrite loom (and optional sibling h5ad) scFAIR fields.

Fixes:
  - row_attrs/feature_name (Ensembl-release display rules)
  - attrs/schema_reference -> scFAIR schema URL
  - attrs/schema_version -> 7.1.0

  python fix_feature_name.v8.py -i /path/output.loom [--h5ad /path/output.h5ad] \\
    --dburl postgres:5434/asap_data_v8 [--schema-only] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import urlparse

import h5py
import numpy as np

from ensembl_release_names import (
    compute_feature_names,
    subdomain_from_ensembl_database,
)

SCFAIR_SCHEMA_VERSION = "7.1.0"
SCFAIR_SCHEMA_REFERENCE = (
    "https://github.com/scFAIR/scFAIR/blob/main/schema/7.1.0/schema.md"
)


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_attr(value.item())
        return [_decode_attr(v) for v in value.tolist()]
    return value


def _read_1d_strings(node) -> list[str]:
    if isinstance(node, h5py.Dataset):
        raw = node[()]
        if isinstance(raw, np.ndarray):
            items = raw.tolist()
        else:
            items = [raw]
        out = []
        for v in items:
            if isinstance(v, (bytes, bytearray)):
                out.append(v.decode("utf-8", "replace"))
            else:
                out.append(str(v))
        return out
    if isinstance(node, h5py.Group):
        enc = _decode_attr(node.attrs.get("encoding-type", ""))
        if enc in ("categorical",) and "categories" in node and "codes" in node:
            cats = _read_1d_strings(node["categories"])
            codes = node["codes"][()]
            return [cats[int(c)] if int(c) >= 0 else "" for c in codes]
        if "values" in node:
            return _read_1d_strings(node["values"])
    raise ValueError(f"Unsupported feature vector node: {node}")


def _write_loom_string_vector(hf: h5py.File, path: str, values: list[str]) -> None:
    if path in hf:
        del hf[path]
    arr = np.asarray(values, dtype=object)
    hf.create_dataset(
        path,
        data=arr,
        dtype=h5py.string_dtype("utf-8"),
        compression="gzip",
        compression_opts=4,
    )


def _write_h5ad_feature_name(hf: h5py.File, values: list[str]) -> None:
    """Replace var/feature_name as a categorical column (AnnData-compatible)."""
    path = "var/feature_name"
    if path in hf:
        del hf[path]

    # Preserve category order of first occurrence.
    categories: list[str] = []
    index_of: dict[str, int] = {}
    codes = np.empty(len(values), dtype=np.int32)
    for i, v in enumerate(values):
        if v not in index_of:
            index_of[v] = len(categories)
            categories.append(v)
        codes[i] = index_of[v]

    g = hf.create_group(path)
    g.attrs["encoding-type"] = "categorical"
    g.attrs["encoding-version"] = "0.2.0"
    g.attrs["ordered"] = False
    g.create_dataset(
        "categories",
        data=np.asarray(categories, dtype=object),
        dtype=h5py.string_dtype("utf-8"),
    )
    g.create_dataset("codes", data=codes)

    # Ensure column-order lists feature_name.
    co = hf["var"].attrs.get("column-order")
    if co is not None:
        cols = [_decode_attr(x) for x in (co.tolist() if isinstance(co, np.ndarray) else list(co))]
        if "feature_name" not in cols:
            cols.append("feature_name")
            hf["var"].attrs["column-order"] = np.asarray(cols, dtype=object)


def _parse_dburl(dburl: str) -> tuple[str, int, str]:
    u = urlparse("http://" + dburl) if "://" not in dburl else urlparse(dburl)
    host = u.hostname
    port = u.port or 5432
    dbname = (u.path or "").lstrip("/")
    if not host or not dbname:
        raise ValueError(f"Invalid --dburl {dburl!r}; expected HOST:PORT/DBNAME")
    return host, int(port), dbname


def lookup_ensembl_db_name(*, dburl: str, tax_id: int) -> tuple[str, str]:
    """Return (ensembl_db_name, subdomain) for NCBI tax_id."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    if not user or not password:
        raise ValueError("POSTGRES_USER and POSTGRES_PASSWORD are required")

    host, port, dbname = _parse_dburl(dburl)
    sql = """
        SELECT o.ensembl_db_name AS ensembl_db_name,
               s.name AS subdomain
        FROM organisms o
        LEFT JOIN ensembl_subdomains s ON s.id = o.ensembl_subdomain_id
        WHERE o.tax_id = %s
        LIMIT 1
    """
    with psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (tax_id,))
            row = cur.fetchone()
    if not row or not row.get("ensembl_db_name"):
        raise ValueError(f"No organisms.ensembl_db_name for tax_id={tax_id}")
    subdomain = (row.get("subdomain") or "").strip().lower()
    if not subdomain:
        raise ValueError(f"No ensembl_subdomain for tax_id={tax_id}")
    return str(row["ensembl_db_name"]).strip(), subdomain


def tax_id_from_term(term: str) -> int:
    s = str(term or "").strip()
    if s.upper().startswith("NCBITAXON:"):
        tail = s.split(":", 1)[1]
        if tail.isdigit():
            return int(tail)
    if s.isdigit():
        return int(s)
    raise ValueError(f"Cannot parse organism_ontology_term_id={term!r}")


def _write_loom_scalar(hf: h5py.File, path: str, value) -> None:
    if path in hf:
        del hf[path]
    if isinstance(value, str):
        hf.create_dataset(path, data=value, dtype=h5py.string_dtype("utf-8"))
    else:
        hf.create_dataset(path, data=value)


def _read_scalar_attr(attrs_group, key: str):
    if key not in attrs_group:
        return None
    return _decode_attr(attrs_group[key][()])


def _write_h5ad_uns_value(hf: h5py.File, key: str, value) -> None:
    """Write a scalar uns value with AnnData-compatible encoding attrs."""
    path = f"uns/{key}"
    if "uns" not in hf:
        hf.create_group("uns")
    if path in hf:
        del hf[path]

    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        ds = hf.create_dataset(path, data=value, dtype=h5py.string_dtype("utf-8"))
        ds.attrs["encoding-type"] = "string"
        ds.attrs["encoding-version"] = "0.2.0"
        return
    if isinstance(value, (bool, np.bool_)):
        ds = hf.create_dataset(path, data=bool(value))
        ds.attrs["encoding-type"] = "numeric"
        ds.attrs["encoding-version"] = "0.2.0"
        return
    if isinstance(value, (int, np.integer)):
        ds = hf.create_dataset(path, data=int(value))
        ds.attrs["encoding-type"] = "numeric"
        ds.attrs["encoding-version"] = "0.2.0"
        return
    if isinstance(value, (float, np.floating)):
        ds = hf.create_dataset(path, data=float(value))
        ds.attrs["encoding-type"] = "numeric"
        ds.attrs["encoding-version"] = "0.2.0"
        return
    # Fallback: stringify
    ds = hf.create_dataset(path, data=str(value), dtype=h5py.string_dtype("utf-8"))
    ds.attrs["encoding-type"] = "string"
    ds.attrs["encoding-version"] = "0.2.0"


def sync_loom_attrs_to_h5ad_uns(loom_path: str, h5ad_path: str) -> int:
    """Copy loom /attrs/* into h5ad uns (required for scFAIR uns.* checks)."""
    with h5py.File(loom_path, "r") as loom:
        if "attrs" not in loom:
            return 0
        items = []
        for key in loom["attrs"].keys():
            items.append((key, _decode_attr(loom["attrs"][key][()])))

    # Always enforce scFAIR schema identity after sync.
    items = [
        (k, v)
        for k, v in items
        if k not in ("schema_reference", "schema_version")
    ]
    items.append(("schema_reference", SCFAIR_SCHEMA_REFERENCE))
    items.append(("schema_version", SCFAIR_SCHEMA_VERSION))

    with h5py.File(h5ad_path, "a") as hf:
        for key, value in items:
            # Skip empty keys; keep nested structures out of this simple sync.
            if isinstance(value, (list, dict)):
                continue
            _write_h5ad_uns_value(hf, key, value)
    return len(items)


def fix_h5ad(
    h5ad_path: str,
    *,
    loom_path: str | None,
    names: list[str] | None,
    dry_run: bool,
    schema_only: bool,
) -> dict:
    with h5py.File(h5ad_path, "r") as hf:
        old_fn = (
            _read_1d_strings(hf["var/feature_name"])
            if (not schema_only and "var" in hf and "feature_name" in hf["var"])
            else None
        )
        old_ref = None
        old_ver = None
        uns_keys = list(hf["uns"].keys()) if "uns" in hf else []
        if "uns" in hf:
            if "schema_reference" in hf["uns"]:
                old_ref = _decode_attr(hf["uns/schema_reference"][()])
            if "schema_version" in hf["uns"]:
                old_ver = _decode_attr(hf["uns/schema_version"][()])
        if not schema_only and names is not None and old_fn is not None:
            if len(old_fn) != len(names):
                raise ValueError(
                    f"h5ad n_vars={len(old_fn)} != loom feature_name length "
                    f"{len(names)} for {h5ad_path}"
                )

    fn_changed = (
        0
        if (schema_only or names is None or old_fn is None)
        else sum(1 for a, b in zip(old_fn, names) if a != b)
    )
    report = {
        "h5ad": h5ad_path,
        "feature_name_changed": fn_changed,
        "schema_reference_changed": old_ref != SCFAIR_SCHEMA_REFERENCE,
        "schema_version_changed": old_ver != SCFAIR_SCHEMA_VERSION,
        "uns_keys_before": uns_keys,
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    with h5py.File(h5ad_path, "a") as hf:
        if not schema_only and names is not None:
            _write_h5ad_feature_name(hf, names)
    if loom_path:
        n = sync_loom_attrs_to_h5ad_uns(loom_path, h5ad_path)
        report["uns_attrs_synced"] = n
        print(f"Synced {n} loom /attrs keys into h5ad uns", flush=True)
    else:
        with h5py.File(h5ad_path, "a") as hf:
            _write_h5ad_uns_value(hf, "schema_reference", SCFAIR_SCHEMA_REFERENCE)
            _write_h5ad_uns_value(hf, "schema_version", SCFAIR_SCHEMA_VERSION)
    return report


def fix_loom(
    loom_path: str,
    *,
    dburl: str | None,
    ensembl_db_name: str | None,
    ensembl_subdomain: str | None,
    dry_run: bool,
    schema_only: bool,
) -> dict:
    with h5py.File(loom_path, "r") as hf:
        attrs = hf["attrs"]
        old_ref = _read_scalar_attr(attrs, "schema_reference")
        old_ver = _read_scalar_attr(attrs, "schema_version")
        release = int(_decode_attr(attrs["ensembl_release"][()])) if not schema_only else None
        ensembl_database = (
            str(_decode_attr(attrs["ensembl_database"][()])) if not schema_only else None
        )
        tax_term = (
            str(_decode_attr(attrs["organism_ontology_term_id"][()]))
            if not schema_only
            else None
        )
        accessions = (
            _read_1d_strings(hf["row_attrs/Accession"]) if not schema_only else []
        )
        if not schema_only and "row_attrs/feature_biotype" in hf:
            biotypes = _read_1d_strings(hf["row_attrs/feature_biotype"])
        else:
            biotypes = ["gene"] * len(accessions)
        old_fn = (
            _read_1d_strings(hf["row_attrs/feature_name"])
            if (not schema_only and "row_attrs/feature_name" in hf)
            else None
        )

    schema_ref_changed = old_ref != SCFAIR_SCHEMA_REFERENCE
    schema_ver_changed = old_ver != SCFAIR_SCHEMA_VERSION

    report = {
        "loom": loom_path,
        "schema_reference_old": old_ref,
        "schema_reference_changed": schema_ref_changed,
        "schema_version_old": old_ver,
        "schema_version_changed": schema_ver_changed,
        "dry_run": dry_run,
        "schema_only": schema_only,
        "feature_name_changed": 0,
        "n_genes": 0,
    }

    new_fn = None
    if not schema_only:
        if ensembl_db_name and ensembl_subdomain:
            db_name = ensembl_db_name.strip()
            subdomain = ensembl_subdomain.strip().lower()
        elif dburl:
            tax_id = tax_id_from_term(tax_term)
            db_name, subdomain = lookup_ensembl_db_name(dburl=dburl, tax_id=tax_id)
        else:
            raise ValueError(
                "Provide --ensembl-db-name and --ensembl-subdomain, or --dburl"
            )

        label_sub = subdomain_from_ensembl_database(ensembl_database)
        if label_sub and label_sub != subdomain:
            raise ValueError(
                f"ensembl_database={ensembl_database!r} maps to {label_sub!r} but "
                f"organisms subdomain is {subdomain!r}"
            )
        if not subdomain:
            subdomain = label_sub
        if not subdomain or not db_name:
            raise ValueError(
                f"Cannot resolve ensembl_db_name/subdomain for ensembl_database={ensembl_database!r}"
            )

        new_fn = compute_feature_names(
            accessions=accessions,
            biotypes=biotypes,
            ensembl_subdomain=subdomain,
            ensembl_db_name=db_name,
            release=release,
        )
        changed = (
            len(new_fn)
            if old_fn is None
            else sum(1 for a, b in zip(old_fn, new_fn) if a != b)
        )
        report.update(
            {
                "n_genes": len(new_fn),
                "ensembl_release": release,
                "ensembl_db_name": db_name,
                "subdomain": subdomain,
                "feature_name_changed": changed,
                "had_feature_name": old_fn is not None,
            }
        )

    if dry_run:
        return report

    with h5py.File(loom_path, "a") as hf:
        if not schema_only and new_fn is not None:
            _write_loom_string_vector(hf, "row_attrs/feature_name", new_fn)
        _write_loom_scalar(hf, "attrs/schema_reference", SCFAIR_SCHEMA_REFERENCE)
        _write_loom_scalar(hf, "attrs/schema_version", SCFAIR_SCHEMA_VERSION)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fix loom/h5ad feature_name and schema_reference for scFAIR"
    )
    parser.add_argument("-i", "--input", required=True, help="Absolute path to output.loom")
    parser.add_argument("--h5ad", default=None, help="Optional sibling .h5ad to update")
    parser.add_argument(
        "--dburl",
        required=False,
        default=None,
        help="ASAP gene DB HOST:PORT/DBNAME (optional if --ensembl-db-name given)",
    )
    parser.add_argument(
        "--ensembl-db-name",
        default=None,
        help="organisms.ensembl_db_name (preferred over --dburl lookup)",
    )
    parser.add_argument(
        "--ensembl-subdomain",
        default=None,
        help="ensembl_subdomains.name e.g. vertebrates (preferred over --dburl lookup)",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only rewrite schema_reference/schema_version (skip feature_name)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    loom = args.input
    if not os.path.isfile(loom):
        print(f"ERROR loom not found: {loom}", file=sys.stderr)
        return 1
    if not args.schema_only:
        has_names = bool(args.ensembl_db_name and args.ensembl_subdomain)
        if not has_names and not args.dburl:
            print(
                "ERROR need --ensembl-db-name/--ensembl-subdomain or --dburl "
                "(unless --schema-only)",
                file=sys.stderr,
            )
            return 1

    loom_report = fix_loom(
        loom,
        dburl=args.dburl,
        ensembl_db_name=args.ensembl_db_name,
        ensembl_subdomain=args.ensembl_subdomain,
        dry_run=args.dry_run,
        schema_only=args.schema_only,
    )
    print(
        f"loom feature_name_changed={loom_report.get('feature_name_changed', 0)}"
        f"/{loom_report.get('n_genes', 0)} "
        f"schema_reference_changed={loom_report['schema_reference_changed']} "
        f"schema_version_changed={loom_report['schema_version_changed']} "
        f"dry_run={args.dry_run}",
        flush=True,
    )

    if args.h5ad:
        if not os.path.isfile(args.h5ad):
            print(f"ERROR h5ad not found: {args.h5ad}", file=sys.stderr)
            return 1
        names = None
        if not args.schema_only:
            with h5py.File(loom, "r") as hf:
                accessions = _read_1d_strings(hf["row_attrs/Accession"])
                biotypes = (
                    _read_1d_strings(hf["row_attrs/feature_biotype"])
                    if "row_attrs/feature_biotype" in hf
                    else ["gene"] * len(accessions)
                )
                release = int(_decode_attr(hf["attrs/ensembl_release"][()]))
            names = compute_feature_names(
                accessions=accessions,
                biotypes=biotypes,
                ensembl_subdomain=loom_report["subdomain"],
                ensembl_db_name=loom_report["ensembl_db_name"],
                release=int(loom_report["ensembl_release"]),
            )
        h5_report = fix_h5ad(
            args.h5ad,
            loom_path=loom,
            names=names,
            dry_run=args.dry_run,
            schema_only=args.schema_only,
        )
        print(
            f"h5ad feature_name_changed={h5_report['feature_name_changed']} "
            f"schema_reference_changed={h5_report['schema_reference_changed']} "
            f"schema_version_changed={h5_report['schema_version_changed']} "
            f"dry_run={args.dry_run}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

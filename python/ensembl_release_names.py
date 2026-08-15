#!/usr/bin/env python3
"""Ensembl release-scoped gene display names (scFAIR feature_name rules).

Mirrors ASAP EnsemblReleaseGeneNameResolver / update_genes.rake:
display_xref name when assigned, otherwise the Ensembl stable_id.
"""

from __future__ import annotations

import gzip
import os
import re
import tarfile
from pathlib import Path
from typing import Final

ENSEMBL_DB_LABELS: Final[dict[str, str]] = {
    "vertebrates": "Ensembl",
    "bacteria": "Ensembl Bacteria",
    "fungi": "Ensembl Fungi",
    "metazoa": "Ensembl Metazoa",
    "plants": "Ensembl Plants",
    "protists": "Ensembl Protists",
}
ENSEMBL_LABEL_TO_SUBDOMAIN: Final[dict[str, str]] = {
    label: subdomain for subdomain, label in ENSEMBL_DB_LABELS.items()
}

ENSEMBL_GENE_STABLE_ID_COLUMN: Final[int] = 12
ENSEMBL_GENE_DISPLAY_XREF_COLUMN: Final[int] = 7
ENSEMBL_XREF_ID_COLUMN: Final[int] = 0
ENSEMBL_XREF_NAME_COLUMN: Final[int] = 3
DEFAULT_ENSEMBL_DATA_DIR: Final[str] = "/mnt/asap_data/ensembl"

ERCC_PATTERN: Final = re.compile(r"^ERCC[-_](\d+)$", re.IGNORECASE)


def ensembl_data_base_dirs() -> list[Path]:
    dirs: list[Path] = []
    primary = os.environ.get("ENSEMBL_DATA_DIR", DEFAULT_ENSEMBL_DATA_DIR).strip()
    if primary:
        dirs.append(Path(primary))
    merge = os.environ.get("ENSEMBL_MERGE_DATA_DIRS", "").strip()
    if merge:
        for part in re.split(r"[,:]", merge):
            part = part.strip()
            if part:
                dirs.append(Path(part))
    return dirs


def normalize_ensembl_stable_id(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s == "\\N" or s == "__unknown":
        return None
    return re.sub(r"\.\d+$", "", s)


def normalize_ercc_id(query: str | None) -> str | None:
    if not query:
        return None
    match = ERCC_PATTERN.match(str(query).strip())
    if not match:
        return None
    return f"ERCC-{match.group(1)}"


def subdomain_from_ensembl_database(label: str | None) -> str | None:
    if not label:
        return None
    return ENSEMBL_LABEL_TO_SUBDOMAIN.get(str(label).strip())


def _ensembl_gene_name_from_display_xref(
    display_xref_id: str, xref_names: dict[str, str], stable_id: str
) -> str:
    name = xref_names.get(str(display_xref_id).strip(), "").strip()
    if not name or name == "\\N":
        return stable_id
    cleaned = re.sub(r"\s+\(\s*\d+\s+of\s+\w+\s*\)", "", name).strip()
    return cleaned or stable_id


def _ensembl_ensure_table_file(
    organism_dir: Path, release_dir: Path, ensembl_db_name: str, table_name: str
) -> Path | None:
    path = organism_dir / table_name
    if path.is_file() and path.stat().st_size > 0:
        return path

    gz_path = organism_dir / f"{table_name}.gz"
    if gz_path.is_file():
        organism_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz_path, "rb") as src, open(path, "wb") as dst:
            dst.write(src.read())
        if path.is_file() and path.stat().st_size > 0:
            return path

    archive = release_dir / f"{ensembl_db_name}.tgz"
    if not archive.is_file():
        return None

    organism_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = [
            m
            for m in tar.getmembers()
            if m.isfile()
            and (
                m.name.endswith("/" + table_name)
                or m.name.endswith("/" + table_name + ".gz")
            )
        ]
        if not members:
            return None
        member = members[0]
        extracted = tar.extractfile(member)
        if extracted is None:
            return None
        raw = extracted.read()
        if member.name.endswith(".gz"):
            raw = gzip.decompress(raw)
        out_path = organism_dir / table_name
        with open(out_path, "wb") as dst:
            dst.write(raw)
        return out_path if out_path.is_file() and out_path.stat().st_size > 0 else None


def load_ensembl_release_gene_names(
    *,
    ensembl_subdomain: str,
    ensembl_db_name: str,
    release: int,
    ensembl_ids: list[str],
) -> dict[str, str]:
    """Map Ensembl stable_id -> display gene_name for one release (else stable_id)."""
    subdomain = (ensembl_subdomain or "").strip().lower()
    db_name = (ensembl_db_name or "").strip()
    if not subdomain or not db_name or int(release) <= 0:
        raise ValueError(
            "Cannot resolve feature_name: ensembl_subdomain, ensembl_db_name, "
            "and ensembl_release are required"
        )

    target_ids = {normalize_ensembl_stable_id(i) for i in ensembl_ids}
    target_ids.discard(None)
    if not target_ids:
        return {}

    release_dir = None
    for base in ensembl_data_base_dirs():
        candidate = base / subdomain / str(int(release))
        if candidate.is_dir():
            release_dir = candidate
            break
    if release_dir is None:
        searched = ", ".join(str(p) for p in ensembl_data_base_dirs()) or "(none)"
        raise ValueError(
            f"Cannot resolve feature_name: Ensembl release directory not found for "
            f"{subdomain}/{int(release)} under {searched}. "
            f"Set ENSEMBL_DATA_DIR (default {DEFAULT_ENSEMBL_DATA_DIR})."
        )

    organism_dir = release_dir / db_name
    gene_path = _ensembl_ensure_table_file(organism_dir, release_dir, db_name, "gene.txt")
    xref_path = _ensembl_ensure_table_file(organism_dir, release_dir, db_name, "xref.txt")
    if gene_path is None or xref_path is None:
        raise ValueError(
            f"Cannot resolve feature_name: gene.txt/xref.txt missing for "
            f"{subdomain}/{int(release)}/{db_name} under {release_dir}"
        )

    xref_names: dict[str, str] = {}
    with open(xref_path, "r", encoding="iso-8859-1", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= ENSEMBL_XREF_NAME_COLUMN:
                continue
            xref_names[parts[ENSEMBL_XREF_ID_COLUMN]] = parts[ENSEMBL_XREF_NAME_COLUMN]

    names: dict[str, str] = {}
    with open(gene_path, "r", encoding="iso-8859-1", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= ENSEMBL_GENE_STABLE_ID_COLUMN:
                continue
            stable_id = normalize_ensembl_stable_id(parts[ENSEMBL_GENE_STABLE_ID_COLUMN])
            if not stable_id or stable_id not in target_ids:
                continue
            display_xref_id = parts[ENSEMBL_GENE_DISPLAY_XREF_COLUMN]
            names[stable_id] = _ensembl_gene_name_from_display_xref(
                display_xref_id, xref_names, stable_id
            )
    return names


def compute_feature_names(
    *,
    accessions: list[str],
    biotypes: list[str] | None,
    ensembl_subdomain: str,
    ensembl_db_name: str,
    release: int,
) -> list[str]:
    """Build scFAIR feature_name vector aligned to accessions."""
    if biotypes is None:
        biotypes = ["gene"] * len(accessions)
    if len(biotypes) != len(accessions):
        raise ValueError(
            f"biotypes length {len(biotypes)} != accessions length {len(accessions)}"
        )

    gene_ids = []
    for acc, bt in zip(accessions, biotypes):
        if str(bt) == "spike-in" or (acc and str(acc).upper().startswith("ERCC")):
            continue
        sid = normalize_ensembl_stable_id(acc)
        if sid:
            gene_ids.append(sid)

    release_names = load_ensembl_release_gene_names(
        ensembl_subdomain=ensembl_subdomain,
        ensembl_db_name=ensembl_db_name,
        release=int(release),
        ensembl_ids=gene_ids,
    )

    out: list[str] = []
    for acc, bt in zip(accessions, biotypes):
        if str(bt) == "spike-in" or (acc and str(acc).upper().startswith("ERCC")):
            ercc = normalize_ercc_id(acc) or str(acc).strip()
            out.append(f"{ercc} (spike-in control)")
            continue
        stable = normalize_ensembl_stable_id(acc)
        if not stable:
            out.append(str(acc or "").strip() or "unknown")
            continue
        out.append(release_names.get(stable, stable))
    return out

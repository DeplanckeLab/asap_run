#!/usr/bin/env python3
"""Ensembl release-scoped gene display names (scFAIR feature_name rules).

Mirrors ASAP EnsemblReleaseGeneNameResolver / update_genes.rake:
display_xref name when assigned, otherwise the Ensembl stable_id.

When gene.txt/xref.txt are missing or empty for the requested release
(e.g. Ensembl published empty xref dumps), the nearest available next or
previous release under ENSEMBL_DATA_DIR is used for name resolution.
Tables are read in-memory (or from existing files); the Ensembl tree is
never written — asap_run mounts it read-only.
"""

from __future__ import annotations

import gzip
import io
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


def _available_releases(subdomain: str) -> list[int]:
    releases: set[int] = set()
    for base in ensembl_data_base_dirs():
        sub_dir = base / subdomain
        if not sub_dir.is_dir():
            continue
        for child in sub_dir.iterdir():
            if child.is_dir() and child.name.isdigit():
                releases.add(int(child.name))
    return sorted(releases)


def _releases_by_distance(preferred: int, available: list[int]) -> list[int]:
    """Preferred first, then nearest neighbors; on ties prefer next over previous."""
    if not available:
        return []
    ordered = sorted(
        available,
        key=lambda release: (
            abs(release - preferred),
            0 if release >= preferred else 1,
            release,
        ),
    )
    return ordered


def _resolve_release_dir(subdomain: str, release: int) -> Path | None:
    for base in ensembl_data_base_dirs():
        candidate = base / subdomain / str(int(release))
        if candidate.is_dir():
            return candidate
    return None


def _read_table_bytes(
    organism_dir: Path, release_dir: Path, ensembl_db_name: str, table_name: str
) -> bytes | None:
    """Return non-empty table contents without writing into the Ensembl tree."""
    path = organism_dir / table_name
    if path.is_file() and path.stat().st_size > 0:
        return path.read_bytes()

    gz_path = organism_dir / f"{table_name}.gz"
    if gz_path.is_file() and gz_path.stat().st_size > 0:
        with gzip.open(gz_path, "rb") as src:
            raw = src.read()
        return raw if raw else None

    archive = release_dir / f"{ensembl_db_name}.tgz"
    if not archive.is_file():
        return None

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
        return raw if raw else None


def _load_names_from_release_dir(
    *,
    release_dir: Path,
    ensembl_db_name: str,
    target_ids: set[str],
) -> dict[str, str] | None:
    organism_dir = release_dir / ensembl_db_name
    gene_raw = _read_table_bytes(organism_dir, release_dir, ensembl_db_name, "gene.txt")
    xref_raw = _read_table_bytes(organism_dir, release_dir, ensembl_db_name, "xref.txt")
    if gene_raw is None or xref_raw is None:
        return None

    xref_names: dict[str, str] = {}
    with io.TextIOWrapper(
        io.BytesIO(xref_raw), encoding="iso-8859-1", errors="replace"
    ) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= ENSEMBL_XREF_NAME_COLUMN:
                continue
            xref_names[parts[ENSEMBL_XREF_ID_COLUMN]] = parts[ENSEMBL_XREF_NAME_COLUMN]

    names: dict[str, str] = {}
    with io.TextIOWrapper(
        io.BytesIO(gene_raw), encoding="iso-8859-1", errors="replace"
    ) as fh:
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


def resolve_ensembl_release_gene_names(
    *,
    ensembl_subdomain: str,
    ensembl_db_name: str,
    release: int,
    ensembl_ids: list[str],
) -> tuple[dict[str, str], int]:
    """Map stable_id -> display name; may use a neighbor release if dumps are empty.

    Returns (names, used_release).
    """
    subdomain = (ensembl_subdomain or "").strip().lower()
    db_name = (ensembl_db_name or "").strip()
    preferred = int(release)
    if not subdomain or not db_name or preferred <= 0:
        raise ValueError(
            "Cannot resolve feature_name: ensembl_subdomain, ensembl_db_name, "
            "and ensembl_release are required"
        )

    target_ids = {normalize_ensembl_stable_id(i) for i in ensembl_ids}
    target_ids.discard(None)
    if not target_ids:
        return {}, preferred

    available = _available_releases(subdomain)
    candidates = _releases_by_distance(preferred, available)
    if not candidates:
        searched = ", ".join(str(p) for p in ensembl_data_base_dirs()) or "(none)"
        raise ValueError(
            f"Cannot resolve feature_name: no Ensembl release directories for "
            f"{subdomain} under {searched}. "
            f"Set ENSEMBL_DATA_DIR (default {DEFAULT_ENSEMBL_DATA_DIR})."
        )

    tried: list[int] = []
    for candidate in candidates:
        release_dir = _resolve_release_dir(subdomain, candidate)
        if release_dir is None:
            continue
        tried.append(candidate)
        names = _load_names_from_release_dir(
            release_dir=release_dir,
            ensembl_db_name=db_name,
            target_ids=target_ids,
        )
        if names is not None:
            return names, candidate

    raise ValueError(
        f"Cannot resolve feature_name: gene.txt/xref.txt missing or empty for "
        f"{subdomain}/{db_name} at release {preferred} and neighbors "
        f"(tried {tried})"
    )


def load_ensembl_release_gene_names(
    *,
    ensembl_subdomain: str,
    ensembl_db_name: str,
    release: int,
    ensembl_ids: list[str],
) -> dict[str, str]:
    """Map Ensembl stable_id -> display gene_name (neighbor fallback if needed)."""
    names, _used_release = resolve_ensembl_release_gene_names(
        ensembl_subdomain=ensembl_subdomain,
        ensembl_db_name=ensembl_db_name,
        release=release,
        ensembl_ids=ensembl_ids,
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

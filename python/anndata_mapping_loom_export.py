#!/usr/bin/env python3
"""Apply Loom /attrs/anndata_mapping when building AnnData / H5AD.

The mapping is the single source of truth for matrix roles (X / raw / layers),
index keys, and obsm/varm paths. Writers: ASAP AnndataMappingPersistService.
Spec: docs/loom-creation-spec-for-h5ad-roundtrip.md
"""

from __future__ import annotations

import html
import json
from typing import Any

import h5py
import numpy as np


ATTR_NAME = "anndata_mapping"
ATTR_PATH = f"/attrs/{ATTR_NAME}"


def _decode_scalar(raw: Any) -> Any:
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        return bytes(raw).decode("utf-8")
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            return _decode_scalar(raw.item())
        if raw.ndim == 1 and raw.size == 1:
            return _decode_scalar(raw[0])
        return [_decode_scalar(v) for v in raw.tolist()]
    if isinstance(raw, np.generic):
        return raw.item()
    return raw


def _decode_attr_values(raw):
    a = np.asanyarray(raw)
    scalar = a.ndim == 0
    if scalar:
        a = np.reshape(a, (1,))
    if a.size and (np.issubdtype(a.dtype, np.number) or np.issubdtype(a.dtype, np.bool_)):
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


def load_anndata_mapping(loom_file: str) -> dict[str, Any]:
    """Load /attrs/anndata_mapping JSON from a Loom file. Raises if missing/invalid."""
    with h5py.File(loom_file, "r") as hf:
        raw = None
        if "attrs" in hf and ATTR_NAME in hf["attrs"]:
            raw = hf["attrs"][ATTR_NAME][()]
        elif ATTR_NAME in hf.attrs:
            raw = hf.attrs[ATTR_NAME]
        if raw is None:
            raise ValueError(
                f"Loom is missing {ATTR_PATH}. Refresh anndata_mapping before export."
            )
        text = _decode_scalar(raw)
        if not isinstance(text, str):
            text = str(text)
        try:
            mapping = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{ATTR_PATH} is not valid JSON: {exc}") from exc
        if not isinstance(mapping, dict):
            raise ValueError(f"{ATTR_PATH} must be a JSON object")
        if not mapping.get("x_path"):
            raise ValueError(f"{ATTR_PATH} is missing required key x_path")
        return mapping


def resolve_dataset(hf: h5py.File, path: str):
    p = path if path.startswith("/") else f"/{path}"
    if p not in hf:
        raise ValueError(f"Mapped loom path not found: {path}")
    node = hf[p]
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"Mapped loom path is not a dataset: {path}")
    return node


def read_matrix_csr_cells_by_genes(dataset, start: int | None = None, end: int | None = None):
    """Read loom genes x cells (slice) as cells x genes float32 CSR."""
    from scipy import sparse

    if start is None and end is None:
        block = dataset[:, :]
    else:
        block = dataset[:, start:end]
    if hasattr(block, "toarray"):
        block = block.toarray()
    block = np.asarray(block, dtype=np.float32, order="C")
    return sparse.csr_matrix(block.T)


def _as_1d_strings(values) -> list[str]:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D names, got shape {arr.shape}")
    out: list[str] = []
    for v in arr:
        if isinstance(v, (bytes, bytearray)):
            out.append(html.unescape(v.decode("utf-8", errors="replace")))
        else:
            out.append(html.unescape(str(v)))
    return out


def _pandas_index_name_for_h5ad(h5ad_index_key: str | None) -> str | None:
    """Map anndata_mapping h5ad_*_index_key to pandas Index.name for AnnData write.

    AnnData stores an unnamed index as the dataset ``_index`` (CellXGene default).
    A non-``_index`` name becomes that dataset name and must then appear in
    column-order for ASAP scFAIR checks.
    """
    key = (h5ad_index_key or "_index").strip() or "_index"
    if key == "_index":
        return None
    return key


def build_obs_var_obsm_varm(hf: h5py.File, mapping: dict[str, Any], n_genes: int, n_cells: int):
    """Build obs/var/obsm/varm from mapping (+ 1D attrs under obs_path/var_path)."""
    import pandas as pd

    obs_path = mapping.get("obs_path") or "/col_attrs"
    var_path = mapping.get("var_path") or "/row_attrs"
    obs_index_key = mapping.get("obs_index_key") or "CellID"
    var_index_key = mapping.get("var_index_key") or "Accession"
    h5ad_obs_index_key = mapping.get("h5ad_obs_index_key") or "_index"
    h5ad_var_index_key = mapping.get("h5ad_var_index_key") or "_index"
    obsm_map = mapping.get("obsm") if isinstance(mapping.get("obsm"), dict) else {}
    varm_map = mapping.get("varm") if isinstance(mapping.get("varm"), dict) else {}
    categoricals = mapping.get("categoricals") if isinstance(mapping.get("categoricals"), dict) else {}

    obs_group = hf[obs_path] if obs_path in hf else None
    var_group = hf[var_path] if var_path in hf else None
    if obs_group is None:
        raise ValueError(f"Mapped obs_path not found: {obs_path}")
    if var_group is None:
        raise ValueError(f"Mapped var_path not found: {var_path}")

    if obs_index_key not in obs_group:
        raise ValueError(f"Loom is missing {obs_path}/{obs_index_key} (obs index)")
    if var_index_key not in var_group:
        raise ValueError(f"Loom is missing {var_path}/{var_index_key} (var index)")

    obs_index = _as_1d_strings(_decode_attr_values(obs_group[obs_index_key][()]))
    var_index = _as_1d_strings(_decode_attr_values(var_group[var_index_key][()]))
    if len(obs_index) != n_cells:
        raise ValueError(f"{obs_index_key} length {len(obs_index)} != n_cells {n_cells}")
    if len(var_index) != n_genes:
        raise ValueError(f"{var_index_key} length {len(var_index)} != n_genes {n_genes}")
    if len(set(var_index)) != n_genes:
        raise ValueError(
            f"{var_index_key} values are not unique ({len(set(var_index))} unique / {n_genes})"
        )

    reserved_obs = {obs_index_key, "obs_names", "cell_names", "CellName"}
    reserved_var = {var_index_key, "var_names", "gene_ids"}
    mapped_obsm_paths = {str(p) for p in obsm_map.values()}
    mapped_varm_paths = {str(p) for p in varm_map.values()}

    obs_data: dict[str, Any] = {}
    for key in obs_group.keys():
        path = f"{obs_path}/{key}"
        if key in reserved_obs or path in mapped_obsm_paths:
            continue
        vals = np.asarray(_decode_attr_values(obs_group[key][()]))
        if vals.ndim == 2:
            # Unmapped 2D col attr: only include if listed in obsm map.
            continue
        if vals.ndim != 1:
            raise ValueError(f"Unsupported {path} ndim={vals.ndim}")
        obs_data[key] = vals

    var_data: dict[str, Any] = {}
    for key in var_group.keys():
        path = f"{var_path}/{key}"
        if key in reserved_var or path in mapped_varm_paths:
            continue
        vals = np.asarray(_decode_attr_values(var_group[key][()]))
        if vals.ndim != 1:
            continue
        var_data[key] = vals

    # Loom supplies values via obs_index_key/var_index_key; H5AD index dataset
    # names come from h5ad_*_index_key (default _index, matching CellXGene).
    obs = pd.DataFrame(
        obs_data,
        index=pd.Index(obs_index, name=_pandas_index_name_for_h5ad(h5ad_obs_index_key)),
    )
    var = pd.DataFrame(
        var_data,
        index=pd.Index(var_index, name=_pandas_index_name_for_h5ad(h5ad_var_index_key)),
    )

    for key, cat_meta in categoricals.items():
        if key not in obs.columns and key not in var.columns:
            continue
        cats = cat_meta.get("categories") if isinstance(cat_meta, dict) else None
        ordered = bool(cat_meta.get("ordered")) if isinstance(cat_meta, dict) else False
        if not cats:
            continue
        target = obs if key in obs.columns else var
        target[key] = pd.Categorical(target[key].astype(str), categories=[str(c) for c in cats], ordered=ordered)

    obsm: dict[str, np.ndarray] = {}
    for key, path in obsm_map.items():
        ds = resolve_dataset(hf, str(path))
        arr = np.asarray(ds[()])
        if arr.ndim != 2:
            raise ValueError(f"obsm[{key}] at {path} must be 2D, got ndim={arr.ndim}")
        if arr.shape[0] == n_cells:
            obsm[str(key)] = arr
        elif arr.shape[1] == n_cells:
            obsm[str(key)] = arr.T
        else:
            raise ValueError(
                f"obsm[{key}] at {path} shape {arr.shape} does not match n_cells={n_cells}"
            )

    varm: dict[str, np.ndarray] = {}
    for key, path in varm_map.items():
        ds = resolve_dataset(hf, str(path))
        arr = np.asarray(ds[()])
        if arr.ndim != 2:
            raise ValueError(f"varm[{key}] at {path} must be 2D, got ndim={arr.ndim}")
        if arr.shape[0] == n_genes:
            varm[str(key)] = arr
        elif arr.shape[1] == n_genes:
            varm[str(key)] = arr.T
        else:
            raise ValueError(
                f"varm[{key}] at {path} shape {arr.shape} does not match n_genes={n_genes}"
            )

    return obs, var, obsm, varm


def matrix_shape(hf: h5py.File, path: str) -> tuple[int, int]:
    ds = resolve_dataset(hf, path)
    if ds.ndim != 2:
        raise ValueError(f"Matrix at {path} must be 2D, got shape {ds.shape}")
    return int(ds.shape[0]), int(ds.shape[1])


def copy_loom_attrs_to_uns(loom_file: str, adata, mapping: dict[str, Any] | None = None) -> int:
    """Copy loom /attrs/* into adata.uns as decoded scalars / JSON text.

    Keys listed in mapping['uns_json_keys'] (e.g. analysis_pipeline) stay as JSON
    strings. Expanding them to nested dicts breaks anndata write_elem on typical
    pipeline payloads (list-of-dicts with nulls / mixed parameter value types:
    "Can't implicitly convert non-string objects to strings"). Consumers that
    need structure should json.loads the string.

    ``mapping`` is accepted for call-site compatibility; JSON keys are not expanded.
    """
    _ = mapping

    n = 0
    with h5py.File(loom_file, "r") as hf:
        if "attrs" not in hf:
            return 0
        for key in hf["attrs"].keys():
            raw = hf["attrs"][key][()]
            adata.uns[key] = _decode_scalar(raw)
            n += 1
    if n:
        print(f"Copied {n} loom /attrs keys into uns", flush=True)
    return n


def apply_mapping_matrices_to_adata(adata, loom_file: str, mapping: dict[str, Any]) -> None:
    """Replace adata.X / raw / layers according to anndata_mapping (after a naive read_loom)."""
    import anndata as ad

    x_path = str(mapping["x_path"])
    raw_x_path = mapping.get("raw_x_path")
    raw_x_path = str(raw_x_path) if raw_x_path else None
    layers_map = mapping.get("layers") if isinstance(mapping.get("layers"), dict) else {}

    with h5py.File(loom_file, "r") as hf:
        n_genes, n_cells = matrix_shape(hf, x_path)
        if adata.n_obs != n_cells or adata.n_vars != n_genes:
            # read_loom used /matrix shape; remap if x_path differs but dims match loom.
            n_genes_m, n_cells_m = matrix_shape(hf, "/matrix") if "/matrix" in hf else (n_genes, n_cells)
            if adata.n_obs != n_cells_m or adata.n_vars != n_genes_m:
                raise ValueError(
                    f"AnnData shape {adata.shape} does not match loom matrices "
                    f"(x_path={x_path} -> cells={n_cells}, genes={n_genes})"
                )
            n_genes, n_cells = n_genes_m, n_cells_m

        X = read_matrix_csr_cells_by_genes(resolve_dataset(hf, x_path))
        if X.shape != (n_cells, n_genes) and X.shape == (adata.n_obs, adata.n_vars):
            pass
        elif X.shape != (adata.n_obs, adata.n_vars):
            # Prefer mapped X even when naive read_loom used /matrix of same dims.
            if X.shape[0] != adata.n_obs or X.shape[1] != adata.n_vars:
                raise ValueError(
                    f"Mapped X from {x_path} has shape {X.shape}, expected {adata.shape}"
                )

        adata.X = X

        new_layers: dict[str, Any] = {}
        used = {x_path}
        if raw_x_path:
            used.add(raw_x_path)
            raw_X = read_matrix_csr_cells_by_genes(resolve_dataset(hf, raw_x_path))
            if raw_X.shape != adata.X.shape:
                raise ValueError(
                    f"raw_x_path {raw_x_path} shape {raw_X.shape} != X shape {adata.X.shape}"
                )
            adata.raw = ad.AnnData(X=raw_X, var=adata.var.copy())

        for layer_name, layer_path in layers_map.items():
            layer_path = str(layer_path)
            if layer_path in used:
                continue
            layer_X = read_matrix_csr_cells_by_genes(resolve_dataset(hf, layer_path))
            if layer_X.shape != adata.X.shape:
                raise ValueError(
                    f"layers[{layer_name}] at {layer_path} shape {layer_X.shape} "
                    f"!= X shape {adata.X.shape}"
                )
            new_layers[str(layer_name)] = layer_X

        adata.layers.clear()
        for k, v in new_layers.items():
            adata.layers[k] = v

    print(
        f"Applied anndata_mapping matrices: X={x_path}"
        + (f" raw={raw_x_path}" if raw_x_path else "")
        + f" layers={list(new_layers.keys())}",
        flush=True,
    )


def apply_mapping_embeddings_to_adata(adata, loom_file: str, mapping: dict[str, Any]) -> None:
    """Set obsm/varm from mapping paths (overrides naive read_loom placement)."""
    with h5py.File(loom_file, "r") as hf:
        n_genes = int(adata.n_vars)
        n_cells = int(adata.n_obs)
        _obs, _var, obsm, varm = build_obs_var_obsm_varm(hf, mapping, n_genes, n_cells)

    # Keep obs/var from read_loom (already UTF-8 recovered); only refresh embeddings.
    adata.obsm.clear()
    for k, v in obsm.items():
        adata.obsm[k] = v
    if hasattr(adata, "varm"):
        adata.varm.clear()
        for k, v in varm.items():
            adata.varm[k] = v

    # Apply categoricals onto existing columns when declared.
    categoricals = mapping.get("categoricals") if isinstance(mapping.get("categoricals"), dict) else {}
    if categoricals:
        import pandas as pd

        for key, cat_meta in categoricals.items():
            cats = cat_meta.get("categories") if isinstance(cat_meta, dict) else None
            ordered = bool(cat_meta.get("ordered")) if isinstance(cat_meta, dict) else False
            if not cats:
                continue
            if key in adata.obs.columns:
                adata.obs[key] = pd.Categorical(
                    adata.obs[key].astype(str),
                    categories=[str(c) for c in cats],
                    ordered=ordered,
                )
            elif key in adata.var.columns:
                adata.var[key] = pd.Categorical(
                    adata.var[key].astype(str),
                    categories=[str(c) for c in cats],
                    ordered=ordered,
                )

    print(
        f"Applied anndata_mapping embeddings: obsm={list(obsm.keys())} varm={list(varm.keys())}",
        flush=True,
    )


def reindex_from_mapping(adata, loom_file: str, mapping: dict[str, Any]) -> None:
    """Ensure obs/var index values match the loom mapping; H5AD names use h5ad_* keys."""
    import pandas as pd

    obs_index_key = mapping.get("obs_index_key") or "CellID"
    var_index_key = mapping.get("var_index_key") or "Accession"
    h5ad_obs_index_key = mapping.get("h5ad_obs_index_key") or "_index"
    h5ad_var_index_key = mapping.get("h5ad_var_index_key") or "_index"
    obs_path = mapping.get("obs_path") or "/col_attrs"
    var_path = mapping.get("var_path") or "/row_attrs"

    with h5py.File(loom_file, "r") as hf:
        obs_ds = resolve_dataset(hf, f"{obs_path}/{obs_index_key}")
        var_ds = resolve_dataset(hf, f"{var_path}/{var_index_key}")
        obs_index = _as_1d_strings(_decode_attr_values(obs_ds[()]))
        var_index = _as_1d_strings(_decode_attr_values(var_ds[()]))

    if len(obs_index) != adata.n_obs:
        raise ValueError(f"{obs_index_key} length {len(obs_index)} != n_obs {adata.n_obs}")
    if len(var_index) != adata.n_vars:
        raise ValueError(f"{var_index_key} length {len(var_index)} != n_vars {adata.n_vars}")

    adata.obs_names = pd.Index(
        obs_index, name=_pandas_index_name_for_h5ad(h5ad_obs_index_key)
    )
    adata.var_names = pd.Index(
        var_index, name=_pandas_index_name_for_h5ad(h5ad_var_index_key)
    )
    if obs_index_key in adata.obs.columns:
        adata.obs.drop(columns=[obs_index_key], inplace=True)
    if var_index_key in adata.var.columns:
        adata.var.drop(columns=[var_index_key], inplace=True)


def apply_anndata_mapping(adata, loom_file: str, mapping: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full post-read_loom application of anndata_mapping. Returns the mapping used."""
    mapping = mapping or load_anndata_mapping(loom_file)
    reindex_from_mapping(adata, loom_file, mapping)
    apply_mapping_matrices_to_adata(adata, loom_file, mapping)
    apply_mapping_embeddings_to_adata(adata, loom_file, mapping)
    copy_loom_attrs_to_uns(loom_file, adata, mapping)
    return mapping

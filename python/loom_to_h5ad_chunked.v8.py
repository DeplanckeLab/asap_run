#!/usr/bin/env python3
"""Chunked Loom -> H5AD converter with bounded RAM (AnnData-native encoding).

Does not replace loom_to_h5ad.v8.py. Streams cell chunks into on-disk CSR
scratch, then writes the h5ad shell (obs/var/uns/obsm/varm) and streams CSR
arrays (X, raw/X, layers/*) from scratch into the file. Peak RAM stays near
one chunk plus metadata — not a full sparse X at assemble/write time.

Uses /attrs/anndata_mapping for matrix roles (X / raw / layers) and indices.

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


def require_feature_name_from_loom(adata) -> None:
    """feature_name must come from parse (Ensembl-release display rules), not genes.name."""
    if "feature_name" not in adata.var.columns:
        raise ValueError(
            "Loom is missing row_attrs/feature_name. Re-parse with a current parse.v8.py "
            "so feature_name is written from the Ensembl release dumps."
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
    obs = pd.DataFrame(obs_data, index=pd.Index(obs_index, name=None))

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
    var = pd.DataFrame(var_data, index=pd.Index(var_index, name=None))

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


def _h5_copy_dataset_chunked(
    src,
    dst_group,
    name: str,
    *,
    dtype,
    compression: str | None = "gzip",
    chunk_elems: int = 2_000_000,
) -> None:
    """Copy a 1D HDF5 dataset in bounded slices (no full-array materialization)."""
    import numpy as np

    n = int(src.shape[0])
    if n == 0:
        chunks = (1024,)
    else:
        chunks = (min(chunk_elems, n),)
    kwargs: dict = {
        "shape": (n,),
        "maxshape": (None,),
        "dtype": dtype,
        "chunks": chunks,
    }
    if compression is not None:
        kwargs["compression"] = compression
    dst = dst_group.create_dataset(name, **kwargs)
    if n == 0:
        return
    for start in range(0, n, chunk_elems):
        end = min(start + chunk_elems, n)
        dst[start:end] = np.asarray(src[start:end], dtype=dtype)


def _stream_csr_group_from_scratch(
    scratch_group,
    h5ad_path: str,
    key: str,
    n_rows: int,
    n_cols: int,
    *,
    compression: str | None = "gzip",
) -> None:
    """Write an AnnData CSR group by streaming scratch datasets (bounded RAM)."""
    import h5py

    if int(scratch_group.attrs["n_rows_filled"]) != n_rows:
        raise ValueError(
            f"CSR row mismatch: filled={scratch_group.attrs['n_rows_filled']} "
            f"expected={n_rows}"
        )

    parts = [p for p in key.split("/") if p]
    if not parts:
        raise ValueError(f"Invalid CSR key: {key!r}")

    with h5py.File(h5ad_path, "a") as out:
        parent = out
        for part in parts[:-1]:
            if part not in parent:
                raise ValueError(
                    f"Missing parent group {part!r} while writing {key!r}; "
                    "create encoding attrs before streaming CSR"
                )
            parent = parent[part]
        name = parts[-1]
        if name in parent:
            del parent[name]
        g = parent.create_group(name)
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = [int(n_rows), int(n_cols)]
        _h5_copy_dataset_chunked(
            scratch_group["data"], g, "data", dtype="float32", compression=compression
        )
        _h5_copy_dataset_chunked(
            scratch_group["indices"],
            g,
            "indices",
            dtype="int32",
            compression=compression,
        )
        _h5_copy_dataset_chunked(
            scratch_group["indptr"],
            g,
            "indptr",
            dtype="int64",
            compression=compression,
        )


def _h5ad_has_matrix_x(h5ad_file: str) -> bool:
    """True when the file has a real AnnData X (CSR group or dense dataset)."""
    import h5py

    if not os.path.isfile(h5ad_file) or os.path.getsize(h5ad_file) <= 0:
        return False
    try:
        with h5py.File(h5ad_file, "r") as f:
            if "X" not in f:
                return False
            x = f["X"]
            if isinstance(x, h5py.Dataset):
                return x.size > 0
            # CSR / CSC encoding
            return "data" in x and "indices" in x and "indptr" in x
    except OSError:
        return False


def _write_h5ad_streamed_from_scratch(
    h5ad_file: str,
    scratch_path: str,
    *,
    obs,
    var,
    obsm: dict,
    varm: dict,
    loom_file: str,
    mapping: dict,
    n_cells: int,
    n_genes: int,
    has_raw: bool,
    layer_names: list[str],
) -> None:
    """Write metadata via write_elem, then stream CSR matrices from scratch.

    Writes to ``h5ad_file + ".partial"`` and renames only after X is present, so a
    failed mid-write cannot leave a metadata-only file that looks "ready".
    """
    import anndata as ad
    import h5py
    import numpy as np
    from anndata.io import write_elem
    from anndata_mapping_loom_export import (
        copy_loom_attr_groups_to_h5ad_uns,
        copy_loom_attrs_to_uns,
    )
    from scipy import sparse

    ad.settings.allow_write_nullable_strings = True
    partial_file = f"{h5ad_file}.partial"
    if os.path.exists(partial_file):
        os.remove(partial_file)

    # Placeholder X is never written; only obs/var/uns/obsm/varm go through write_elem.
    shell = ad.AnnData(
        X=sparse.csr_matrix((n_cells, n_genes), dtype=np.float32),
        obs=obs,
        var=var,
    )
    for key, arr in obsm.items():
        shell.obsm[key] = np.asarray(arr)
    for key, arr in varm.items():
        shell.varm[key] = np.asarray(arr)
    copy_loom_attrs_to_uns(loom_file, shell, mapping)
    require_feature_name_from_loom(shell)

    try:
        dataset_kwargs = {"compression": "gzip"}
        with h5py.File(partial_file, "w") as f:
            f.attrs["encoding-type"] = "anndata"
            f.attrs["encoding-version"] = "0.1.0"
            write_elem(f, "obs", shell.obs, dataset_kwargs=dataset_kwargs)
            write_elem(f, "var", shell.var, dataset_kwargs=dataset_kwargs)
            write_elem(f, "uns", dict(shell.uns), dataset_kwargs=dataset_kwargs)
            copy_loom_attr_groups_to_h5ad_uns(loom_file, f, mapping)
            if len(shell.obsm):
                write_elem(f, "obsm", dict(shell.obsm), dataset_kwargs=dataset_kwargs)
            if len(shell.varm):
                write_elem(f, "varm", dict(shell.varm), dataset_kwargs=dataset_kwargs)
            if has_raw:
                raw_g = f.create_group("raw")
                raw_g.attrs["encoding-type"] = "raw"
                raw_g.attrs["encoding-version"] = "0.1.0"
                # Match previous behavior: raw.var is a copy of var.
                write_elem(raw_g, "var", shell.var, dataset_kwargs=dataset_kwargs)
            if layer_names:
                layers_g = f.create_group("layers")
                layers_g.attrs["encoding-type"] = "dict"
                layers_g.attrs["encoding-version"] = "0.1.0"

        del shell
        gc.collect()

        with h5py.File(scratch_path, "r") as scratch:
            print(f"Streaming CSR X -> {partial_file}", flush=True)
            _stream_csr_group_from_scratch(
                scratch["X"], partial_file, "X", n_cells, n_genes, compression="gzip"
            )
            if has_raw:
                print(f"Streaming CSR raw/X -> {partial_file}", flush=True)
                _stream_csr_group_from_scratch(
                    scratch["extra___raw__"],
                    partial_file,
                    "raw/X",
                    n_cells,
                    n_genes,
                    compression="gzip",
                )
            for name in layer_names:
                print(f"Streaming CSR layers[{name!r}] -> {partial_file}", flush=True)
                _stream_csr_group_from_scratch(
                    scratch[f"extra_{name}"],
                    partial_file,
                    f"layers/{name}",
                    n_cells,
                    n_genes,
                    compression="gzip",
                )

        if not _h5ad_has_matrix_x(partial_file):
            raise RuntimeError(
                f"H5AD partial write finished without matrix X: {partial_file}"
            )

        os.replace(partial_file, h5ad_file)
    except Exception:
        if os.path.exists(partial_file):
            os.remove(partial_file)
        raise


def convert_chunked(
    loom_file: str,
    h5ad_file: str,
    *,
    chunk_cells: int,
    tmp_parent: str,
) -> tuple[int, int, dict]:
    import h5py
    from anndata_mapping_loom_export import (
        build_obs_var_obsm_varm,
        load_anndata_mapping,
        matrix_shape,
        read_matrix_csr_cells_by_genes,
        resolve_dataset,
    )

    install_loompy_none_attr_recovery()

    mapping = load_anndata_mapping(loom_file)
    x_path = str(mapping["x_path"])
    raw_x_path = mapping.get("raw_x_path")
    raw_x_path = str(raw_x_path) if raw_x_path else None
    layers_map = mapping.get("layers") if isinstance(mapping.get("layers"), dict) else {}

    with h5py.File(loom_file, "r") as hf:
        n_genes, n_cells = matrix_shape(hf, x_path)
        obs, var, obsm, varm = build_obs_var_obsm_varm(hf, mapping, n_genes, n_cells)

    if chunk_cells <= 0:
        raise ValueError("chunk_cells must be positive")

    print(
        f"Using anndata_mapping x_path={x_path} raw_x_path={raw_x_path} "
        f"layers={list(layers_map.keys())}",
        flush=True,
    )

    # Extra matrices to stream: declared layers + raw (not X).
    extra_paths: list[tuple[str, str]] = []
    used = {x_path}
    if raw_x_path and raw_x_path not in used:
        extra_paths.append(("__raw__", raw_x_path))
        used.add(raw_x_path)
    for layer_name, layer_path in layers_map.items():
        layer_path = str(layer_path)
        if layer_path in used:
            continue
        extra_paths.append((str(layer_name), layer_path))
        used.add(layer_path)

    tmp_dir = tempfile.mkdtemp(prefix="loom2h5ad_chunks_", dir=tmp_parent)
    scratch_path = os.path.join(tmp_dir, "csr_scratch.h5")
    try:
        with h5py.File(scratch_path, "w") as scratch:
            x_scratch = _init_csr_scratch(scratch, "X", n_cells)
            extra_scratches: dict[str, object] = {}

            with h5py.File(loom_file, "r") as hf:
                x_ds = resolve_dataset(hf, x_path)
                extra_dss = {
                    name: resolve_dataset(hf, path) for name, path in extra_paths
                }
                for name, _path in extra_paths:
                    extra_scratches[name] = _init_csr_scratch(scratch, f"extra_{name}", n_cells)

                n_chunks = (n_cells + chunk_cells - 1) // chunk_cells
                for chunk_i, start in enumerate(range(0, n_cells, chunk_cells)):
                    end = min(start + chunk_cells, n_cells)
                    print(
                        f"Chunk {chunk_i + 1}/{n_chunks}: cells [{start}:{end}) "
                        f"({end - start} cells)",
                        flush=True,
                    )
                    X = read_matrix_csr_cells_by_genes(x_ds, start, end)
                    _append_csr_chunk(x_scratch, X)
                    for name, ds in extra_dss.items():
                        block = read_matrix_csr_cells_by_genes(ds, start, end)
                        _append_csr_chunk(extra_scratches[name], block)
                        del block
                    del X
                    gc.collect()

        has_raw = "__raw__" in extra_scratches
        layer_names = [name for name in extra_scratches if name != "__raw__"]

        print(f"Writing h5ad (streamed CSR) -> {h5ad_file}", flush=True)
        _write_h5ad_streamed_from_scratch(
            h5ad_file,
            scratch_path,
            obs=obs,
            var=var,
            obsm=obsm,
            varm=varm,
            loom_file=loom_file,
            mapping=mapping,
            n_cells=n_cells,
            n_genes=n_genes,
            has_raw=has_raw,
            layer_names=layer_names,
        )
        gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return n_cells, n_genes, mapping


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
        required=False,
        default=None,
        help="Unused (kept for StdMethod CLI compatibility). feature_name comes from the loom.",
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
        n_obs, n_vars, mapping = convert_chunked(
            loom_file,
            h5ad_file,
            chunk_cells=args.chunk_cells,
            tmp_parent=output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - surface any convert failure in output.json
        for path in (h5ad_file, f"{h5ad_file}.partial"):
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        fail(f"Loom to H5AD conversion failed: {exc}", output_json)

    if not os.path.isfile(h5ad_file) or os.path.getsize(h5ad_file) <= 0:
        fail("H5AD output was not written or is empty", output_json)
    if not _h5ad_has_matrix_x(h5ad_file):
        try:
            os.remove(h5ad_file)
        except OSError:
            pass
        fail("H5AD output is missing matrix X (incomplete write)", output_json)

    payload = {
        "status": "success",
        "input_loom": loom_file,
        "output_h5ad": h5ad_file,
        "output_h5ad_bytes": os.path.getsize(h5ad_file),
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
        "chunk_cells": int(args.chunk_cells),
        "converter": "chunked_anndata_mapping_streamed_csr",
        "anndata_mapping_x_path": mapping.get("x_path"),
        "anndata_mapping_raw_x_path": mapping.get("raw_x_path"),
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

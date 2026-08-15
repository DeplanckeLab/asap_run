#!/usr/bin/env python3
"""Chunked Loom -> H5AD converter with bounded RAM (AnnData-native encoding).

Does not replace loom_to_h5ad.v8.py. This variant streams cell chunks into
temporary AnnData files, then stacks sparse X / obs / obsm / layers and writes
the final .h5ad with AnnData.write_h5ad so encoding stays AnnData-compliant.
Peak RAM is roughly one matrix chunk during conversion, then the stacked sparse
CSR at assemble time (still far below a dense full-loom load).

Mirrors anndata.io.read_loom conventions used by ASAP looms:
  - loom matrix is genes x cells; AnnData X is cells x genes (float32 CSR)
  - CellID -> obs index; Gene -> var index
  - 1D col/row attrs -> obs/var columns; 2D col attrs -> obsm
  - named loom layer "X" -> adata.layers["X"] when present

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
        obs_index = [str(i) for i in range(n_cells)]
    if len(obs_index) != n_cells:
        raise ValueError(f"CellID length {len(obs_index)} != n_cells {n_cells}")
    obs = pd.DataFrame(obs_data, index=pd.Index(obs_index, name="CellID"))

    var_data: dict = {}
    for key in ds.ra.keys():
        vals = np.asarray(ds.ra[key])
        if vals.ndim != 1:
            raise ValueError(f"Unsupported ra[{key!r}] ndim={vals.ndim}")
        if key in {"Gene", "var_names", "gene_names", "GeneName"}:
            continue
        var_data[key] = vals

    if "Gene" in ds.ra:
        var_index = _as_1d_object_strings(ds.ra["Gene"])
    else:
        var_index = [str(i) for i in range(n_genes)]
    if len(var_index) != n_genes:
        raise ValueError(f"Gene length {len(var_index)} != n_genes {n_genes}")
    var = pd.DataFrame(var_data, index=pd.Index(var_index, name="Gene"))

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


def convert_chunked(
    loom_file: str,
    h5ad_file: str,
    *,
    chunk_cells: int,
    tmp_parent: str,
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
    chunk_paths: list[str] = []
    try:
        with h5py.File(loom_file, "r") as hf:
            if "matrix" not in hf:
                raise ValueError("Loom file has no /matrix dataset")
            matrix_ds = hf["matrix"]
            layer_x_ds = hf["layers"]["X"] if "layers" in hf and "X" in hf["layers"] else None

            n_chunks = (n_cells + chunk_cells - 1) // chunk_cells
            for chunk_i, start in enumerate(range(0, n_cells, chunk_cells)):
                end = min(start + chunk_cells, n_cells)
                print(
                    f"Chunk {chunk_i + 1}/{n_chunks}: cells [{start}:{end}) "
                    f"({end - start} cells)",
                    flush=True,
                )
                X = read_matrix_chunk_csr(matrix_ds, start, end)
                adata_chunk = ad.AnnData(
                    X=X,
                    obs=obs.iloc[start:end].copy(deep=True),
                    var=var.copy(deep=True),
                )
                for key, arr in obsm.items():
                    adata_chunk.obsm[key] = np.asarray(arr[start:end])
                if layer_x_ds is not None:
                    adata_chunk.layers["X"] = read_matrix_chunk_csr(layer_x_ds, start, end)

                chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_i:05d}.h5ad")
                adata_chunk.write_h5ad(chunk_path)
                chunk_paths.append(chunk_path)
                del adata_chunk, X
                gc.collect()

        if not chunk_paths:
            raise ValueError("No chunks written")

        print(f"Assembling {len(chunk_paths)} chunks -> {h5ad_file}", flush=True)
        if os.path.exists(h5ad_file):
            os.remove(h5ad_file)
        # Avoid anndata.concat here: ASAP looms often have duplicate Gene names,
        # and concat reindexes var (InvalidIndexError). Chunks share identical var
        # order, so stack X/obs/obsm/layers and reuse var from the first chunk.
        from scipy import sparse
        import pandas as pd

        x_parts = []
        obs_parts = []
        obsm_parts: dict[str, list] = {}
        layer_parts: dict[str, list] = {}
        var = None
        for path in chunk_paths:
            piece = ad.read_h5ad(path)
            if var is None:
                var = piece.var.copy(deep=True)
            x_parts.append(piece.X.tocsr())
            obs_parts.append(piece.obs)
            for key in piece.obsm.keys():
                obsm_parts.setdefault(key, []).append(np.asarray(piece.obsm[key]))
            for key in piece.layers.keys():
                layer_parts.setdefault(key, []).append(piece.layers[key].tocsr())
            del piece

        merged = ad.AnnData(
            X=sparse.vstack(x_parts, format="csr"),
            obs=pd.concat(obs_parts, axis=0),
            var=var,
        )
        for key, parts in obsm_parts.items():
            merged.obsm[key] = np.concatenate(parts, axis=0)
        for key, parts in layer_parts.items():
            merged.layers[key] = sparse.vstack(parts, format="csr")
        del x_parts, obs_parts, obsm_parts, layer_parts
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

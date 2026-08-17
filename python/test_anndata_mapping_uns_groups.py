#!/usr/bin/env python3
"""uns_groups copies loom /attrs/spatial as an HDF5 group into h5ad uns/spatial."""

from __future__ import annotations

import os
import tempfile
import types
import unittest

import h5py
import numpy as np

from anndata_mapping_loom_export import (
    copy_loom_attr_groups_to_h5ad_uns,
    copy_loom_attrs_to_uns,
)


def _write_spatial_loom(path: str) -> None:
    with h5py.File(path, "w") as hf:
        hf.create_dataset("/attrs/title", data="visium example")
        spatial = hf.create_group("/attrs/spatial")
        spatial.create_dataset("is_single", data=True)
        lib = spatial.create_group("libA")
        images = lib.create_group("images")
        images.create_dataset("hires", data=np.zeros((8, 8, 3), dtype=np.uint8))
        scales = lib.create_group("scalefactors")
        scales.create_dataset("tissue_hires_scalef", data=0.17)
        scales.create_dataset("spot_diameter_fullres", data=50.0)


class UnsGroupsCopyTest(unittest.TestCase):
    def test_scalar_copy_skips_declared_spatial_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loom = os.path.join(tmp, "in.loom")
            _write_spatial_loom(loom)
            adata = types.SimpleNamespace(uns={})
            mapping = {"uns_groups": {"spatial": "/attrs/spatial"}}
            copy_loom_attrs_to_uns(loom, adata, mapping)
            self.assertEqual(adata.uns["title"], "visium example")
            self.assertNotIn("spatial", adata.uns)

    def test_undeclared_spatial_group_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loom = os.path.join(tmp, "in.loom")
            _write_spatial_loom(loom)
            adata = types.SimpleNamespace(uns={})
            with self.assertRaises(ValueError) as ctx:
                copy_loom_attrs_to_uns(loom, adata, {"uns_groups": {}})
            self.assertIn("uns_groups", str(ctx.exception))

    def test_copies_spatial_group_into_h5ad_uns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loom = os.path.join(tmp, "in.loom")
            h5ad = os.path.join(tmp, "out.h5ad")
            _write_spatial_loom(loom)
            with h5py.File(h5ad, "w") as dst:
                uns = dst.create_group("uns")
                uns.attrs["encoding-type"] = "dict"
            mapping = {"uns_groups": {"spatial": "/attrs/spatial"}}
            n = copy_loom_attr_groups_to_h5ad_uns(loom, h5ad, mapping)
            self.assertEqual(n, 1)
            with h5py.File(h5ad, "r") as hf:
                self.assertIn("spatial", hf["uns"])
                spatial = hf["uns/spatial"]
                self.assertTrue(isinstance(spatial, h5py.Group))
                self.assertEqual(bool(spatial["is_single"][()]), True)
                hires = spatial["libA/images/hires"]
                self.assertEqual(hires.dtype, np.uint8)
                self.assertEqual(hires.ndim, 3)


if __name__ == "__main__":
    unittest.main()

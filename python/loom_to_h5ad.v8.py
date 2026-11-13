#!/usr/bin/env python3
"""Loom -> H5AD export entry point.

Kept under this name because the export_h5ad StdMethod runs `python loom_to_h5ad.v8.py -i ... -o ... -d ... --dburl ...`
(asap_web lib/export_h5ad_v8_std_methods.rb). All arguments are passed unchanged to loom_to_h5ad_chunked.v8.py, which
accepts the same CLI (plus --chunk-cells). The ASAP export and the external catalog import therefore produce the same
h5ad, with bounded RAM. That includes looms with groups under /attrs (e.g. Visium spatial), which the previous loompy-based
in-memory converter could not read.
"""
import os
import sys

if __name__ == "__main__":
    chunked = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loom_to_h5ad_chunked.v8.py")
    os.execv(sys.executable, [sys.executable, chunked] + sys.argv[1:])

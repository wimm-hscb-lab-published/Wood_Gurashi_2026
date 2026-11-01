"""Shared pytest setup for the pipelines tests.

Run from the pipelines folder with the mosaic conda env (~/miniconda3/envs/mosaic):
    ~/miniconda3/envs/mosaic/bin/python -m pytest tests
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")  # no GUI windows during tests

# Make DNA_Utils, CNV_Utils, LOH_Pipeline and PRO_Utils importable.
PIPELINES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIPELINES_DIR not in sys.path:
    sys.path.insert(0, PIPELINES_DIR)

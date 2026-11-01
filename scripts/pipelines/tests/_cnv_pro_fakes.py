"""Lightweight stand-ins for missionbio.mosaic assays / samples.

Used by tests/test_cnv_*.py and tests/test_pro_*.py. Only implements the
surface those modules actually touch (layers, row_attrs, col_attrs, ids(),
barcodes(), get_attribute(), get_labels(), add_row_attr(), [rows, cols]
subsetting and copy()).
"""
import copy as _copy

import numpy as np
import pandas as pd


class FakeAssay:
    def __init__(self, layers, barcodes, ids, row_attrs=None, col_attrs=None):
        self.layers = {k: np.asarray(v) for k, v in layers.items()}
        self.row_attrs = {"barcode": np.asarray(barcodes, dtype=object)}
        self.row_attrs.update({k: np.asarray(v) for k, v in (row_attrs or {}).items()})
        self.col_attrs = {"id": np.asarray(ids, dtype=object)}
        self.col_attrs.update({k: np.asarray(v) for k, v in (col_attrs or {}).items()})

    # --- accessors -------------------------------------------------------
    def barcodes(self):
        return np.asarray(self.row_attrs["barcode"])

    def ids(self):
        return np.asarray(self.col_attrs["id"])

    def get_labels(self):
        return np.asarray(self.row_attrs["label"])

    def add_row_attr(self, name, values):
        self.row_attrs[name] = np.asarray(values)

    def get_attribute(self, name, constraint="row+col"):
        if name in self.layers:
            return pd.DataFrame(self.layers[name], index=self.barcodes(), columns=self.ids())
        if name in self.row_attrs:
            return pd.DataFrame(self.row_attrs[name], index=self.barcodes())
        raise KeyError(name)

    def copy(self):
        return _copy.deepcopy(self)

    # --- subsetting ------------------------------------------------------
    @staticmethod
    def _resolve(key, labels):
        if isinstance(key, slice):
            return np.arange(len(labels))[key]
        key = np.asarray(list(key) if not isinstance(key, np.ndarray) else key)
        if key.dtype == bool:
            return np.where(key)[0]
        if np.issubdtype(key.dtype, np.integer):
            return key
        lookup = {str(v): i for i, v in enumerate(labels)}
        return np.array([lookup[str(k)] for k in key], dtype=int)

    def __getitem__(self, key):
        rows, cols = key
        r = self._resolve(rows, self.barcodes())
        c = self._resolve(cols, self.ids())
        new = FakeAssay.__new__(FakeAssay)
        new.layers = {k: v[np.ix_(r, c)] for k, v in self.layers.items()}
        new.row_attrs = {k: np.asarray(v)[r] for k, v in self.row_attrs.items()}
        new.col_attrs = {k: np.asarray(v)[c] for k, v in self.col_attrs.items()}
        return new


class FakeSample:
    def __init__(self, **assays):
        for name, assay in assays.items():
            setattr(self, name, assay)

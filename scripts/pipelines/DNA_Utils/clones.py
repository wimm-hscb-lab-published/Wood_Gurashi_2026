"""Clone-level tables: clone genotype table, per-clone VAF and clone sizes.

Main entry points
-----------------
:func:`clones_to_dataframe`
    Wrap a Mosaic workflow's ``get_clones`` output in a tidy DataFrame.
:func:`get_VAF`
    Variant allele frequency (VAF %) of every variant in every clone.
:func:`get_clones_size`
    Cell count (and optional percentage) per clone, optionally after
    excluding protein cell types.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd

from .utils import drop_scalar_row_attrs

# Horizontal rule used in the verbose banners.
_RULE = "=" * 60


def _make_vprint(verbose: bool):
    """Return a ``print`` that only prints when ``verbose`` is true."""
    def vprint(msg):
        if verbose:
            print(msg)
    return vprint


def _flatten_variants(v) -> list:
    """Turn a variant spec into a flat list of strings (one nesting level is flattened).

    A string becomes ``[v]``; arrays and Series are flattened and stringified;
    lists/tuples/sets have each element flattened the same way (deeper nested
    lists are stringified as a whole); anything else becomes ``[str(v)]``.
    """
    if isinstance(v, str):
        return [v]
    if isinstance(v, np.ndarray):
        return [str(x) for x in v.ravel().tolist()]
    if isinstance(v, pd.Series):
        return [str(x) for x in v.astype(str).tolist()]
    if isinstance(v, (list, tuple, set)):
        out = []
        for x in v:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, np.ndarray):
                out.extend([str(y) for y in x.ravel().tolist()])
            elif isinstance(x, pd.Series):
                out.extend([str(y) for y in x.astype(str).tolist()])
            elif isinstance(x, (list, tuple, set)):
                out.extend([str(y) for y in x])
            else:
                out.append(str(x))
        return out
    return [str(v)]


def _unique_stripped(values) -> list:
    """Strip each value, drop empty strings and ``None``, and de-duplicate keeping order."""
    seen, clean = set(), []
    for s in (x.strip() for x in values if x is not None):
        if s and s not in seen:
            seen.add(s)
            clean.append(s)
    return clean


def clones_to_dataframe(wf, variants, max_ado_score: float = 0.8,
                        min_clone_size: int = 1) -> pd.DataFrame:
    """Call ``wf.get_clones`` and return its rows as a DataFrame with display names.

    Parameters
    ----------
    wf : Mosaic workflow
        Object with ``get_clones(variants=..., max_ado_score=...,
        min_clone_size=...)`` returning ``(rows, columns_meta, _)``.
    variants : str, array-like, Series or (nested) list
        Variant IDs. Flattened one level, stripped, empty strings removed and
        de-duplicated (first occurrence kept) before being passed on.
    max_ado_score : float
        Passed to ``wf.get_clones``.
    min_clone_size : int
        Passed to ``wf.get_clones``.

    Returns
    -------
    pd.DataFrame
        One row per record; columns renamed from ``id`` to ``name`` using the
        column metadata and ordered as in the metadata (columns not in the
        metadata are dropped). Rows whose variant-name column
        (``"Variant\\nname"``) is blank are removed and the index is reset.

    Notes
    -----
    ``None`` entries are stringified to ``"None"`` while flattening, so the
    later ``None`` filter never removes them.
    """
    variants = _unique_stripped(_flatten_variants(variants))

    rows, cols_meta, _ = wf.get_clones(
        variants=variants,
        max_ado_score=max_ado_score,
        min_clone_size=min_clone_size,
    )

    df = pd.DataFrame.from_records(rows)
    id_to_name = {c["id"]: c["name"] for c in cols_meta}
    df.rename(columns=id_to_name, inplace=True)

    ordered_cols = [c["name"] for c in cols_meta if c["name"] in df.columns]
    df = df[ordered_cols]

    vcol = next(
        (c["name"] for c in cols_meta if c["id"] == "Variant\nname"),
        "Variant\nname",
    )
    if vcol in df.columns:
        df = df[df[vcol].astype(str).str.strip() != ""].reset_index(drop=True)

    return df


def get_VAF(
    Sample,
    label_col: str = "label",
    clone_order: list | None = None,
    output_path: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute variant allele frequencies (VAF %) per clone.

    For each distinct value of ``Sample.dna.row_attrs[label_col]`` the dna
    assay is subset to that clone's cells and Mosaic's
    ``get_variant_bulk_metrics`` column ``"VAF by cell count"`` is taken.

    Parameters
    ----------
    Sample : mosaic Sample
        Sample with a dna assay. If the dna assay has no ``NGT_FILTERED``
        layer it is created as the ``NGT`` layer (side effect).
    label_col : str
        Row attribute with clone labels.
    clone_order : list, optional
        Column order; clones not listed are appended in sorted order (and
        reported when ``verbose``).
    output_path : str, optional
        If given, the table is written as CSV (index label ``"Variants"``);
        the parent directory is created.
    verbose : bool
        Print progress.

    Returns
    -------
    df_vaf : pd.DataFrame
        Variants (rows, as in the bulk-metrics index) × clones (columns).
        Without ``clone_order`` the column order follows ``set`` iteration.
    """
    vprint = _make_vprint(verbose)

    vprint("\n" + _RULE)
    vprint("COMPUTING VAF PER CLONE")
    vprint(_RULE)

    if "NGT_FILTERED" not in Sample.dna.layers:
        Sample.dna.layers["NGT_FILTERED"] = Sample.dna.layers["NGT"]
        vprint("  ✓ Created NGT_FILTERED layer")

    bulk_metrics = Sample.dna.get_variant_bulk_metrics(variants=None)
    df_vaf = pd.DataFrame(index=bulk_metrics["VAF by cell count"].index)

    clones = list(set(Sample.dna.row_attrs[label_col]))
    vprint(f"  Variants: {len(df_vaf)}, Clones: {len(clones)}")

    for clone in clones:
        sample_sub = copy.copy(Sample)
        indices = [
            i for i, x in enumerate(sample_sub.dna.row_attrs[label_col])
            if x == clone
        ]
        subset = sample_sub.dna[indices, :]
        clone_metrics = subset.get_variant_bulk_metrics(variants=None)
        df_vaf[clone] = clone_metrics["VAF by cell count"].values
        vprint(f"    • {clone}: {len(indices)} cells")

    if clone_order is not None:
        valid = [c for c in clone_order if c in df_vaf.columns]
        missing = [c for c in df_vaf.columns if c not in valid]
        if missing:
            vprint(f"  ⚠️  Clones not in clone_order: {missing}")
            valid.extend(sorted(missing))
        df_vaf = df_vaf.reindex(columns=valid)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_vaf.to_csv(output_path, index=True, index_label="Variants")
        vprint(f"  ✓ Saved: {output_path}")

    vprint(_RULE + "\n")
    return df_vaf


def get_clones_size(
    Sample,
    label_col: str = "label",
    desired_order: list | None = None,
    exclude_celltypes: list | None = None,
    return_percentage: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Count cells per clone, optionally excluding protein cell types first.

    Parameters
    ----------
    Sample : mosaic Sample
        Sample with a dna assay (and a protein assay if
        ``exclude_celltypes`` is used).
    label_col : str
        Row attribute with clone labels.
    desired_order : list, optional
        Row order of the result; clones not listed are appended in sorted
        order (and reported when ``verbose``). ``Clone`` becomes an ordered
        Categorical.
    exclude_celltypes : list, optional
        Protein labels (``Sample.protein.get_labels()``) whose cells are
        removed before counting. Scalar row attributes are dropped from both
        assays of ``Sample`` (side effect) and a new Mosaic ``Sample`` is built
        from the kept barcodes.
    return_percentage : bool
        Add a ``Percentage`` column (count / total × 100).
    verbose : bool
        Print progress.

    Returns
    -------
    label_counts : pd.DataFrame
        Columns ``Clone``, ``Count`` (and ``Percentage``). Without
        ``desired_order`` rows are in ``value_counts`` order.

    Raises
    ------
    ValueError
        If no cells remain after exclusion.
    """
    vprint = _make_vprint(verbose)

    vprint("\n" + _RULE)
    vprint("CALCULATING CLONE SIZE")
    vprint(_RULE)

    if exclude_celltypes:
        vprint(f"  Excluding: {exclude_celltypes}")
        prot_labels = Sample.protein.get_labels().astype(str)
        prot_barcodes = np.asarray(Sample.protein.barcodes(), dtype=object)
        mask_keep = ~np.isin(prot_labels, exclude_celltypes)
        bars_keep = prot_barcodes[mask_keep]

        if bars_keep.size == 0:
            raise ValueError("No cells remain after exclusion.")

        drop_scalar_row_attrs(Sample.dna)
        drop_scalar_row_attrs(Sample.protein)

        # Imported lazily: only needed for the exclusion path.
        from missionbio.mosaic.sample import Sample as MosaicSample

        dna_sub = copy.copy(Sample.dna)[bars_keep, :]
        protein_sub = copy.copy(Sample.protein)[bars_keep, :]
        Sample_filtered = MosaicSample(dna=dna_sub, protein=protein_sub)
        vprint(f"  Cells after exclusion: {bars_keep.size}")
    else:
        Sample_filtered = Sample

    labels_series = pd.Series(Sample_filtered.dna.row_attrs[label_col])
    label_counts = labels_series.value_counts().reset_index()
    label_counts.columns = ["Clone", "Count"]
    total = label_counts["Count"].sum()

    if return_percentage:
        label_counts["Percentage"] = (label_counts["Count"] / total) * 100

    if desired_order is not None:
        valid = [l for l in desired_order if l in label_counts["Clone"].values]
        missing = [l for l in label_counts["Clone"].values if l not in desired_order]
        if missing:
            vprint(f"  ⚠️  Clones not in desired_order: {missing}")
            valid.extend(sorted(missing))
        label_counts["Clone"] = pd.Categorical(
            label_counts["Clone"], categories=valid, ordered=True
        )
        label_counts = label_counts.sort_values("Clone").reset_index(drop=True)

    vprint(f"  {len(label_counts)} clones, {total} total cells")
    vprint(_RULE + "\n")
    return label_counts

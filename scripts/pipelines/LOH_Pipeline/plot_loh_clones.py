"""
Phase 5 plotting – plot active LOH, unresolved SUB, possible-LOH (FP), and non-LOH clones.

Main entry point: ``plot_loh_clones``. The plot dataframe is built directly
from ``PipelineState`` and drawn with ``CNV_Utils.visualise_variants_by_clone``
(or a caller-supplied function).  Retained FP labels are deliberately a
separate category: they remain visible for review but are not active LOH
sources, targets, or baseline clones in later pipeline phases.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd

from .loh_config import clone_category

if TYPE_CHECKING:
    from .loh_config import LOHConfig, PipelineState

# Category → tag used in the verbose plotting-order listing.
_DISPLAY_TAG = {"LOH": "LOH", "subclonal_LOH": "SUB", "possible_LOH": "PLOH", "non_LOH": "BASE"}


def _labels_in_category(labels_with_cells: dict, category: str, min_cells: int) -> list:
    """Sorted labels of *category* with at least *min_cells* cells."""
    return sorted(
        lbl for lbl, n in labels_with_cells.items()
        if clone_category(lbl) == category and n >= min_cells
    )


def plot_loh_clones(
    state: "PipelineState",
    *,
    cfg: "LOHConfig | None" = None,
    informative_variants: list | None = None,
    cytobands: "pd.DataFrame | None" = None,
    output_dir: str | None = None,
    output_prefix: str = "LOH_clones_AF",
    include_non_loh: bool = False,
    include_possible_loh: bool = True,
    plot_separate_clones: bool = True,
    split_by_cytobands: bool = False,
    save_figures: bool = True,
    save_pdf: bool | None = None,
    visualise_fn=None,
    verbose: bool | None = None,
):
    """Build and visualise a clone-labelled AF dataframe.

    Active LOH clones are always included. ``SUB: ...`` directional-split
    labels and ``FP: ...`` labels are retained as review-only categories: they
    are included by default, carry no minimum-size gate, and are ordered
    between active LOH and baseline non-LOH clones. Baseline
    non-LOH clones are included only when ``include_non_loh=True`` and still
    use ``cfg.min_final_clone_size``.  ``small`` labels are never plotted.

    Parameters
    ----------
    state : PipelineState
        Supplies ``df`` and the current ``clone_to_cells`` mapping.
    cfg : LOHConfig or None
        Defaults to ``state.config``; also supplies defaults for
        *informative_variants*, *cytobands*, *output_dir*, *save_pdf* and
        *verbose* when those are None.
    informative_variants : list or None
        Variants to plot (those absent from ``state.df`` are skipped).
    cytobands : pd.DataFrame or None
        Forwarded to the visualiser.
    output_dir : str or None
        Output folder; required when *save_figures* is True.
    output_prefix : str, default "LOH_clones_AF"
        File-name prefix forwarded to the visualiser.
    include_non_loh : bool, default False
        Also plot baseline non-LOH clones with ≥ ``min_final_clone_size`` cells.
    include_possible_loh : bool, default True
        Also plot SUB and FP clones.
    plot_separate_clones, split_by_cytobands, save_figures : bool
        Forwarded to the visualiser.
    save_pdf : bool or None
        After plotting, also save every open matplotlib figure as PDF (next
        to its PNG when the figure label is an existing file path, else as
        ``<output_dir>/<output_prefix>_fig<N>.pdf``).
    visualise_fn : callable or None
        Defaults to ``CNV_Utils.visualise_variants_by_clone``.
    verbose : bool or None
        Print clone lists, plotting order and saved paths.

    Returns
    -------
    pd.DataFrame or None
        The plotted dataframe (``state.df`` rows of selected clones plus a
        ``label`` column), or None when there is nothing to plot.

    Raises
    ------
    ValueError
        If *informative_variants* is missing/empty or none are in the data,
        or if *save_figures* is True without *output_dir*.
    ImportError
        If *visualise_fn* is None and ``CNV_Utils`` cannot be imported.

    Notes
    -----
    Within each category clones are ordered by decreasing plotted cell count,
    then by label; categories are ordered LOH, SUB, FP, non-LOH.
    """
    if cfg is None:
        cfg = state.config
    if verbose is None:
        verbose = cfg.verbose
    if informative_variants is None:
        informative_variants = getattr(cfg, "informative_variants", None)
    if cytobands is None:
        cytobands = getattr(cfg, "cytobands", None)
    if output_dir is None:
        output_dir = getattr(cfg, "output_dir", None)
    if save_pdf is None:
        save_pdf = getattr(cfg, "save_pdf", True)

    if informative_variants is None or len(informative_variants) == 0:
        raise ValueError(
            "informative_variants is required. Pass it directly or set "
            "cfg.informative_variants."
        )
    if save_figures and output_dir is None:
        raise ValueError(
            "output_dir is required when save_figures=True. Pass it "
            "directly or set cfg.output_dir."
        )

    min_size = int(cfg.min_final_clone_size)
    labels_with_cells = {
        str(lbl): len(cells)
        for lbl, cells in state.clone_to_cells.items()
        if len(cells) > 0
    }
    loh_labels = _labels_in_category(labels_with_cells, "LOH", 1)
    subclonal_labels = _labels_in_category(labels_with_cells, "subclonal_LOH", 1)
    possible_loh_labels = _labels_in_category(labels_with_cells, "possible_LOH", 1)
    non_loh_labels = _labels_in_category(labels_with_cells, "non_LOH", min_size)

    if verbose:
        for category, labels in (
            ("LOH", loh_labels),
            ("SUB", subclonal_labels),
            ("possible LOH", possible_loh_labels),
            ("non-LOH", non_loh_labels),
        ):
            print(f"\n{category} clones in state: {len(labels)}")
            for lbl in labels:
                print(f"  • {lbl}  ({labels_with_cells[lbl]} cells)")

    plot_labels = list(loh_labels)
    if include_possible_loh:
        plot_labels.extend(subclonal_labels)
        plot_labels.extend(possible_loh_labels)
    if include_non_loh:
        plot_labels.extend(non_loh_labels)

    if not plot_labels:
        print("No selected LOH, possible-LOH, or non-LOH clones found in state → nothing to plot.")
        return None

    df_plot = state.df.copy()
    cell_to_clone = {
        str(cell): str(label)
        for label, cells in state.clone_to_cells.items()
        for cell in cells
    }
    df_plot["label"] = df_plot.index.astype(str).map(cell_to_clone)
    df_plot = df_plot.dropna(subset=["label"]).copy()
    df_plot["label"] = df_plot["label"].astype(str)
    df_plot = df_plot.loc[df_plot["label"].isin(plot_labels)].copy()

    if df_plot.empty:
        print("After filtering, no cells remain in the plotting dataframe.")
        return None

    vars_present = [v for v in informative_variants if v in df_plot.columns]
    if not vars_present:
        raise ValueError("None of informative_variants exist in df columns.")
    if verbose and len(vars_present) < len(informative_variants):
        print(f"  ℹ  {len(informative_variants) - len(vars_present)} informative variant(s) not in df (skipped)")

    counts = df_plot["label"].value_counts()

    def _order(labels):
        return sorted(labels, key=lambda x: (-counts.get(x, 0), x))

    loh_order = _order([x for x in plot_labels if clone_category(x) == "LOH"])
    subclonal_order = _order([x for x in plot_labels if clone_category(x) == "subclonal_LOH"])
    possible_order = _order([x for x in plot_labels if clone_category(x) == "possible_LOH"])
    non_loh_order = _order([x for x in plot_labels if clone_category(x) == "non_LOH"])
    desired_order = loh_order + subclonal_order + possible_order + non_loh_order

    if verbose:
        print(f"\n  Cells after filtering: {len(df_plot)} (from {len(state.df)} total)")
        print(f"  Plotting order ({len(desired_order)} clones):")
        for lbl in desired_order:
            print(f"    [{_DISPLAY_TAG[clone_category(lbl)]}] {lbl}  (n={counts.get(lbl, 0)})")

    if visualise_fn is None:
        try:
            import CNV_Utils
            visualise_fn = CNV_Utils.visualise_variants_by_clone
        except ImportError as exc:
            raise ImportError("Could not import CNV_Utils. Pass visualise_fn explicitly.") from exc

    visualise_fn(
        df=df_plot,
        variants_to_plot=vars_present,
        cytobands=cytobands,
        output_dir=output_dir,
        output_prefix=output_prefix,
        label_col="label",
        desired_clone_plotting_order=desired_order,
        verbose=verbose,
        plot_separate_clones=plot_separate_clones,
        split_by_cytobands=split_by_cytobands,
        save_figures=save_figures,
    )

    if save_figures and save_pdf and output_dir is not None:
        import matplotlib.pyplot as plt
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            png_path = fig.get_label()
            if png_path and os.path.isfile(png_path):
                pdf_path = os.path.splitext(png_path)[0] + ".pdf"
            else:
                pdf_path = os.path.join(output_dir, f"{output_prefix}_fig{fig_num}.pdf")
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            if verbose:
                print(f"  📄 Saved PDF: {pdf_path}")

    if verbose:
        print("✅ Plotting complete.")
    return df_plot

"""Germline variant discovery, blacklist filtering, and visualisation."""

import re
import numpy as np
import pandas as pd

from .utils import drop_scalar_row_attrs


def get_real_somatic_variants(
    differential_variants,
    filtered_germline_in_samples,
    sample,
    use_consequence_if_available: bool = True,
) -> np.ndarray:
    """Return differential variants present in ``sample.dna``, not in the
    germline list, and not intronic.

    Parameters
    ----------
    differential_variants : array-like
        Variant IDs (nested OK).
    filtered_germline_in_samples : array-like, set, or None
        Variant IDs to exclude as germline.
    sample : object
        Must have ``sample.dna.col_attrs`` with at least ``'id'`` and
        ideally ``'Protein'`` (optionally ``'Consequence'``).
    use_consequence_if_available : bool
        Use the ``'Consequence'`` column when present.

    Returns
    -------
    np.ndarray[str]
    """
    dv_flat = [str(x).strip() for x in np.ravel(differential_variants)]
    germline_set = (
        set(map(str, np.ravel(filtered_germline_in_samples)))
        if filtered_germline_in_samples is not None
        else set()
    )
    dv_flat = [v for v in dv_flat if v and v not in germline_set]
    if not dv_flat:
        return np.array([], dtype=str)

    attrs = pd.DataFrame(sample.dna.col_attrs).copy()
    if "id" not in attrs.columns:
        raise KeyError("sample.dna.col_attrs must contain an 'id' column.")
    attrs["id"] = attrs["id"].astype(str).str.strip()

    sub = attrs.loc[attrs["id"].isin(dv_flat)].copy()

    if use_consequence_if_available and "Consequence" in sub.columns:
        is_intronic = (
            sub["Consequence"]
            .astype(str)
            .str.contains("intron", case=False, na=False)
        )
    else:
        prot = (
            sub.get("Protein", pd.Series(index=sub.index, dtype=object))
            .astype(str)
            .fillna("")
        )
        is_intronic = prot.str.contains(r"p\.p\.\?", regex=True) | (
            prot.str.strip() == ""
        )

    keep_ids = set(sub.loc[~is_intronic, "id"])
    return np.array([v for v in dv_flat if v in keep_ids], dtype=str)


def visualize_found_germline_variants(
    Sample_T_subset,
    wd: str,
    MBio_batch: str,
    Patient_ID: str,
    variant_ids=None,
    variant_df: pd.DataFrame = None,
    variant_class_col: str = "germline_broad_class",
    variant_classes=None,
    seed_col: str = None,
    output_label: str = None,
    cytobands_path: str = None,
    save_plots: bool = True,
    show_plots: bool = True,
    min_variants_per_chr: int = 2,
) -> pd.DataFrame:
    """
    Visualise selected germline variants with:
      1. VAF histogram
      2. genome-wide stem plot
      3. per-chromosome stem plots
      4. faceted scatter plot

    Parameters
    ----------
    Sample_T_subset
        T-cell subset sample.

    variant_ids : list-like, optional
        Explicit variant IDs to plot.

    variant_df : pd.DataFrame, optional
        All-variant germline dataframe, e.g. all_variant_germline_df.

    variant_class_col : str
        Column in variant_df used for class filtering.

    variant_classes : list-like, optional
        Classes to keep from variant_class_col.

    seed_col : str, optional
        Boolean seed column to filter variant_df, e.g.
        "usable_germline_het_seed", "clean_germline_het_seed",
        or "rescued_germline_het_seed".

    output_label : str, optional
        Label appended to output plot filenames.

    Returns
    -------
    col_attrs_df : pd.DataFrame
        Variant annotations with pseudobulk VAF values.
    """
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from chrov.viz.figure import plot_with_chroms

    def _safe_label(x):
        x = str(x)
        x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
        x = re.sub(r"_+", "_", x).strip("_")
        return x or "selected_variants"

    def _dedupe_keep_order(items):
        seen = set()
        out = []
        for x in items:
            if pd.isna(x):
                continue
            x = str(x)
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _layer_df(assay, layer):
        try:
            df = assay.get_attribute(layer, constraint="row+col")
        except Exception:
            df = assay.layers[layer]

        if not isinstance(df, pd.DataFrame):
            arr = df
            if hasattr(arr, "toarray"):
                arr = arr.toarray()

            row_index = np.asarray(
                assay.row_attrs.get("barcode", np.arange(arr.shape[0]))
            ).astype(str)

            col_index = np.asarray(
                assay.col_attrs.get("id", np.arange(arr.shape[1]))
            ).astype(str)

            df = pd.DataFrame(arr, index=row_index, columns=col_index)

        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        return df

    def _get_col_attr(assay, keys, default=None):
        for key in keys:
            try:
                if key in assay.col_attrs:
                    return np.asarray(assay.col_attrs[key])
            except Exception:
                pass

        n = len(np.asarray(assay.col_attrs["id"]))
        if default is None:
            return np.array([np.nan] * n)
        return np.array([default] * n)

    def _chrom_sort_key(chrom):
        s = str(chrom).replace("chr", "")
        if s.isdigit():
            return (0, int(s))
        if s.upper() == "X":
            return (1, 23)
        if s.upper() == "Y":
            return (1, 24)
        if s.upper() in {"M", "MT"}:
            return (1, 25)
        return (2, s)

    # Prefer the full DNA assay so explicit variant_ids can be selected
    # even if dna_subset contains a previous subset.
    assay = getattr(Sample_T_subset, "dna", None)
    if assay is None:
        assay = getattr(Sample_T_subset, "dna_subset", None)

    if assay is None:
        raise RuntimeError("Could not find Sample_T_subset.dna or Sample_T_subset.dna_subset.")

    all_assay_variants = np.asarray(assay.col_attrs["id"]).astype(str).tolist()

    # ── Resolve variants to plot ──────────────────────────────────────────
    if variant_df is not None:
        if "variant" not in variant_df.columns:
            raise ValueError("variant_df must contain a 'variant' column.")

        vf = variant_df.copy()
        vf["variant"] = vf["variant"].astype(str)

        if variant_classes is not None:
            if variant_class_col not in vf.columns:
                raise ValueError(
                    f"variant_df does not contain variant_class_col='{variant_class_col}'."
                )
            variant_classes = set(str(x) for x in variant_classes)
            vf = vf[vf[variant_class_col].astype(str).isin(variant_classes)]

        if seed_col is not None:
            if seed_col not in vf.columns:
                raise ValueError(f"variant_df does not contain seed_col='{seed_col}'.")
            vf = vf[vf[seed_col].fillna(False).astype(bool)]

        variant_ids_from_df = vf["variant"].tolist()

        if variant_ids is None:
            variant_ids = variant_ids_from_df
        else:
            # Explicit variant_ids is authoritative. variant_df is used only
            # for annotation/metadata and optional filtering, not to expand
            # the plotted variant set.
            requested = set(str(v) for v in variant_ids)
            vf = vf[vf["variant"].astype(str).isin(requested)].copy()
            variant_ids = list(variant_ids)

    if variant_ids is None:
        # Backwards-compatible behaviour: plot all variants in the assay.
        variant_ids = all_assay_variants

    variant_ids = _dedupe_keep_order(variant_ids)

    assay_variant_set = set(all_assay_variants)
    selected_variants = [v for v in variant_ids if v in assay_variant_set]
    missing_variants = [v for v in variant_ids if v not in assay_variant_set]

    if missing_variants:
        print(
            f"  ⚠ {len(missing_variants)} requested variant(s) were not found "
            "in Sample_T_subset.dna and will be skipped."
        )

    if not selected_variants:
        raise RuntimeError("No requested variants were found in Sample_T_subset.dna.")

    if output_label is None:
        if seed_col is not None:
            output_label = seed_col
        elif variant_classes is not None:
            output_label = "_".join(str(x) for x in variant_classes)
        else:
            output_label = "selected_germline_variants"

    output_label = _safe_label(output_label)

    output_dir = (
        f"{wd}/results/per_sample/{MBio_batch}/{Patient_ID}/"
        "dna_cnv/variants/germline_variants"
    )

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("VISUALIZING GERMLINE VARIANTS")
    print("=" * 60)
    print(f"Patient: {Patient_ID}")
    print(f"Variants requested: {len(variant_ids)}")
    print(f"Variants plotted:   {len(selected_variants)}")
    print(f"Output label:       {output_label}")

    # ── AF matrix and pseudobulk VAF ──────────────────────────────────────
    af_df = _layer_df(assay, "AF").astype(float)
    af_df = af_df[selected_variants]

    pseudobulk_vaf = af_df.mean(axis=0, skipna=True)

    # ── Build variant annotation dataframe ────────────────────────────────
    assay_ids = np.asarray(assay.col_attrs["id"]).astype(str)

    meta = pd.DataFrame({
        "variant": assay_ids,
        "gene symbol": _get_col_attr(assay, ["Gene", "gene", "gene symbol", "SYMBOL"]),
        "chromosome": _get_col_attr(assay, ["CHROM", "chromosome", "chr"]),
        "position": _get_col_attr(assay, ["POS", "position", "pos"]),
    })

    meta["variant"] = meta["variant"].astype(str)
    meta = meta[meta["variant"].isin(selected_variants)].copy()

    meta["variant"] = pd.Categorical(
        meta["variant"],
        categories=selected_variants,
        ordered=True,
    )

    meta = meta.sort_values("variant").reset_index(drop=True)
    meta["variant"] = meta["variant"].astype(str)

    meta["VAF"] = meta["variant"].map(pseudobulk_vaf.to_dict()).astype(float)

    meta["gene symbol"] = meta["gene symbol"].fillna("").astype(str)
    meta["chromosome"] = meta["chromosome"].astype(str)
    meta["position"] = pd.to_numeric(meta["position"], errors="coerce")

    meta["Annotation"] = np.where(
        meta["gene symbol"].str.len() > 0,
        meta["gene symbol"] + " (" + meta["variant"] + ")",
        meta["variant"],
    )

    # Bring selected variant_df columns into the returned dataframe.
    if variant_df is not None:
        extra = variant_df.copy()
        extra["variant"] = extra["variant"].astype(str)
        extra = extra.drop_duplicates("variant", keep="first")

        keep_extra_cols = [
            c for c in [
                "variant",
                "germline_broad_class",
                "donor_genotype",
                "clean_germline_het_seed",
                "rescued_germline_het_seed",
                "usable_germline_het_seed",
                "status",
                "review_verdict",
                "pooled_af",
                "median_af",
                "n_het_cells",
                "total_dp",
                "af_deviation",
                "z_robust_sd",
                "is_pathogenic_excluded",
                "is_blacklisted",
                "is_common_resource",
                "is_known_het_whitelist",
            ]
            if c in extra.columns
        ]

        meta = meta.merge(extra[keep_extra_cols], on="variant", how="left")

    col_attrs_df = meta.copy()

    # ── Plot 1: VAF histogram ─────────────────────────────────────────────
    sns.set_style("whitegrid")

    plt.figure(figsize=(8, 6))
    plt.hist(col_attrs_df["VAF"].dropna(), bins=30, alpha=0.6, color="g")
    plt.yscale("log", base=10)
    plt.title(f"Pseudobulk VAFs: {output_label}")
    plt.xlabel("Allele Frequency (%)")
    plt.ylabel("Frequency (Log10)")

    if save_plots:
        p = f"{output_dir}/Pseudobulk_VAFs_T_cells_{output_label}.png"
        plt.savefig(p, dpi=300, bbox_inches="tight")
        plt.savefig(p.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ── Cytobands ─────────────────────────────────────────────────────────
    if cytobands_path is None:
        cytobands_path = f"{wd}/resources/references/cytobands.tsv"

    cytobands = pd.read_table(cytobands_path)

    if "chromosome" not in cytobands.columns:
        cytobands = pd.read_table(cytobands_path, index_col=[0])
        if "chromosome" not in cytobands.columns:
            cytobands = cytobands.reset_index().rename(columns={"index": "chromosome"})

    cytobands["chromosome"] = cytobands["chromosome"].astype(str)

    # ── Plot 2: genome-wide distribution ──────────────────────────────────
    genome_df = col_attrs_df.dropna(subset=["position", "VAF"]).copy()

    if not genome_df.empty:
        plt.figure()
        plot_with_chroms(
            data=genome_df.copy(),
            cytobands=cytobands,
            kind="stem",
            colx="position",
            coly="VAF",
            col_label="Annotation",
            xkind="loci",
            va="bottom",
            chrom_y=0,
            off=2.25,
            offy=-1.25,
            figsize=[60, 4],
            arc=False,
            fig=None,
            kws_seaborn={},
            kws_annot_chroms={},
            kws_annot_labels={},
        )

        if save_plots:
            p = f"{output_dir}/Germline_variants_distribution_{output_label}.png"
            plt.savefig(p, dpi=300, bbox_inches="tight")
            plt.savefig(p.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        print("  ⚠ No variants with valid position/VAF for genome-wide plot.")

    # ── Plot 3: per-chromosome stem plots ─────────────────────────────────
    chromosomes_plotted = 0

    chromosome_counts = genome_df.groupby("chromosome").size()

    for chr_name in sorted(chromosome_counts.index.tolist(), key=_chrom_sort_key):
        if chromosome_counts.loc[chr_name] < min_variants_per_chr:
            continue

        chr_data = genome_df[genome_df["chromosome"] == chr_name].copy()
        chr_cytobands = cytobands[cytobands["chromosome"] == chr_name].copy()

        if chr_cytobands.empty:
            continue

        safe_chr = _safe_label(chr_name)

        plt.figure()
        plot_with_chroms(
            data=chr_data,
            cytobands=chr_cytobands,
            kind="stem",
            colx="position",
            coly="VAF",
            col_label="Annotation",
            xkind="loci",
            va="bottom",
            chrom_y=0,
            off=0.15,
            offy=0.55,
            figsize=[6, 2],
            arc=False,
            fig=None,
            kws_seaborn={},
            kws_annot_chroms={},
            kws_annot_labels={},
        )

        if save_plots:
            p = f"{output_dir}/Germline_variants_distribution_{output_label}_{safe_chr}.png"
            plt.savefig(p, dpi=300, bbox_inches="tight")
            plt.savefig(p.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()

        chromosomes_plotted += 1

    print(f"    ✓ {chromosomes_plotted} per-chromosome plots")

    # ── Plot 4: faceted scatter ───────────────────────────────────────────
    scatter_df = genome_df.copy()

    if not scatter_df.empty:
        scatter_df["chrom_sort"] = scatter_df["chromosome"].map(_chrom_sort_key)
        scatter_df = scatter_df.sort_values(["chrom_sort", "position"])

        chrom_order = sorted(scatter_df["chromosome"].unique(), key=_chrom_sort_key)

        g = sns.FacetGrid(
            scatter_df,
            col="chromosome",
            col_order=chrom_order,
            col_wrap=4,
            sharey=False,
            height=2.5,
            aspect=1.2,
        )

        g.map_dataframe(
            sns.scatterplot,
            x="position",
            y="VAF",
            alpha=0.6,
            color="g",
        )

        g.set(ylim=(0, 100))
        g.set_axis_labels("Genomic Position", "VAF (%)")
        g.fig.suptitle(
            f"Germline Variant VAFs Across Chromosomes: {output_label}",
            y=1.02,
            fontsize=14,
        )

        if save_plots:
            p = f"{output_dir}/Pseudobulk_VAFs_T_cells_{output_label}_across_chrs.png"
            plt.savefig(p, dpi=300, bbox_inches="tight")
            plt.savefig(p.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        print("  ⚠ No variants with valid position/VAF for faceted scatter plot.")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)

    return col_attrs_df


def find_germline_variants(
    Sample,
    keep_labels,
    annotation_df: pd.DataFrame = None,
    annotation_variant_id_col: str = "Variant ID",

    # AF windows after per-variant NGT/DP filtering
    af_min_common: float = 25,
    af_max_common: float = 75,
    af_min_novel: float = 40,
    af_max_novel: float = 60,

    # genotype/cell filters
    max_hom_frac: float = 0.25,
    max_ngt_frac: float = 0.80,
    min_dp: int = 5,
    min_valid_cells: int = 20,
    min_call_rate_after_ngt_dp: float = 0.20,

    # HET-vs-HOM_ALT dominance filter
    require_het_hom_alt_ratio: bool = True,
    min_het_to_hom_alt_ratio: float = 3.0,
    min_het_frac_among_alt_valid: float = 0.75,
    min_het_valid_cells: int = 3,

    # annotation/resource filters
    exclude_pathogenic: bool = True,
    common_variant_paths: list = None,
    known_het_variants: list = None,
    known_het_variant_paths: list = None,
    treat_known_het_as_common: bool = True,
    blacklist_paths: list = None,
    common_requires: str = "either",  # "either" or "both"

    # output/debug
    return_qc: bool = False,
    verbose: bool = True,
):
    """
    Find likely germline HET variants in selected cells.

    Filtering order
    ---------------
    1. Subset to selected protein labels.
    2. For each variant, remove NGT == 3 cells.
    3. Apply DP >= min_dp to the remaining called cells.
    4. Compute AF/genotype summaries only from valid cells.
    5. Exclude Pathogenic / Likely_pathogenic ClinVar variants.
    6. Exclude variants in genomic blacklist intervals.
    7. Split variants into common-resource versus novel:
       - common-resource variants: mean AF between af_min_common and af_max_common.
       - novel variants: mean AF between af_min_novel and af_max_novel.
    8. Require HET calls to dominate HOM_ALT calls:
       - N_HET_valid / N_HOM_ALT_valid >= min_het_to_hom_alt_ratio.
       - N_HET_valid / (N_HET_valid + N_HOM_ALT_valid) >= min_het_frac_among_alt_valid.

    Notes
    -----
    known_het_variants can be your in-memory germline_whitelist list.

    Returns
    -------
    By default:
        filtered_germline_variants, Sample_T_subset

    If return_qc=True:
        filtered_germline_variants, Sample_T_subset, germline_qc
    """

    import copy
    import os
    from bisect import bisect_right

    import numpy as np
    import pandas as pd
    from missionbio.mosaic.sample import Sample as MosaicSample

    def vprint(msg):
        if verbose:
            print(msg)

    def _norm_chrom(x):
        x = str(x).strip()
        if x == "":
            return x
        return x if x.startswith("chr") else f"chr{x}"

    def _read_id_file(path):
        with open(path) as fh:
            return {line.strip() for line in fh if line.strip()}

    def _read_id_files(paths, mode="either"):
        sets = []

        for path in paths:
            try:
                ids = _read_id_file(path)
                sets.append(ids)
                vprint(f"  Loaded {len(ids):,} IDs from {os.path.basename(path)}")
            except Exception as e:
                vprint(f"  ⚠️ Could not read ID file {path}: {e}")

        if len(sets) == 0:
            return set()

        if mode == "both":
            return set.intersection(*sets)

        return set.union(*sets)

    def _flatten_variant_list(x):
        if x is None:
            return set()

        out = set()

        for v in np.ravel(list(x)):
            v = str(v).strip()
            if v:
                out.add(v)

        return out

    def _parse_variant_ids(variant_ids):
        s = pd.Series(list(variant_ids), name="variant").astype(str)

        parsed = s.str.extract(r"^(chr[^:]+):(\d+):([^/]+)/(.+)$")
        parsed.columns = ["chr", "pos", "ref", "alt"]

        out = pd.DataFrame({"variant": s})
        out = pd.concat([out, parsed], axis=1)

        out["chr"] = out["chr"].map(
            lambda x: _norm_chrom(x) if pd.notna(x) else x
        )
        out["pos"] = pd.to_numeric(out["pos"], errors="coerce")

        return out

    def _read_blacklist(path):
        ext = os.path.splitext(path)[1].lower()

        if ext in (".bed", ".tsv", ".txt"):
            raw = pd.read_csv(path, sep="\t", header=None, comment="#")
            if raw.shape[1] < 3:
                raise ValueError(f"Blacklist has fewer than 3 columns: {path}")

            df = raw.iloc[:, :3].copy()
            df.columns = ["chr", "start", "end"]

        else:
            df = pd.read_csv(path)

            rename = {}
            for c in df.columns:
                cl = c.lower().strip()

                if cl in ("chromosome", "chrom", "chr"):
                    rename[c] = "chr"
                elif cl in ("start", "begin", "chromstart"):
                    rename[c] = "start"
                elif cl in ("stop", "end", "chromend"):
                    rename[c] = "end"

            df = df.rename(columns=rename)
            df = df[["chr", "start", "end"]].copy()

        df["chr"] = df["chr"].map(_norm_chrom)
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df = df.dropna(subset=["chr", "start", "end"]).copy()

        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)

        return df

    def _build_intervals(bed_df):
        intervals = {}

        for chrom, sub in bed_df.groupby("chr"):
            sub = sub.sort_values("start")
            intervals[chrom] = (
                sub["start"].to_numpy(),
                sub["end"].to_numpy(),
            )

        return intervals

    def _is_blacklisted(chrom, pos_1based, intervals):
        chrom = _norm_chrom(chrom)

        if chrom not in intervals:
            return False

        starts, ends = intervals[chrom]
        pos0 = int(pos_1based) - 1

        i = bisect_right(starts, pos0) - 1
        return i >= 0 and pos0 < ends[i]

    ############################################################################
    # Defaults
    ############################################################################

    if blacklist_paths is None:
        blacklist_paths = [
            "/project/meadlab/shared/02__Projects/"
            "02__TP53_MDS_Tapestri/Resources/Blacklist/hg19-blacklist.v2.bed",
            "/project/meadlab/shared/02__Projects/"
            "02__TP53_MDS_Tapestri/Resources/Blacklist/"
            "hg19_blacklist_MergedList_Anshul Kundaje_Stanford.csv",
        ]

    if common_variant_paths is None:
        common_variant_paths = [
            "/project/meadlab/shared/02__Projects/02b__Tapestri/resources/"
            "common_mutations/dbsnp_v155_hg19_common_vars.txt",
            "/project/meadlab/shared/02__Projects/02b__Tapestri/resources/"
            "common_mutations/gnomad_genomes_r2.1.1_hg19_common_vars.txt",
        ]

    if common_requires not in {"either", "both"}:
        raise ValueError("common_requires must be either 'either' or 'both'.")

    vprint("\n" + "=" * 60)
    vprint("FINDING GERMLINE HET-LIKE VARIANTS")
    vprint("=" * 60)

    ############################################################################
    # 1. Subset to selected cells
    ############################################################################

    prot_labels = Sample.protein.get_labels().astype(str)
    prot_barcodes = np.asarray(Sample.protein.barcodes(), dtype=object)

    mask_keep = np.isin(prot_labels, keep_labels)
    bars_keep = prot_barcodes[mask_keep]

    vprint(f"\n  Total cells: {prot_barcodes.size}")
    vprint(f"  Cells kept:  {bars_keep.size}")

    if bars_keep.size == 0:
        raise ValueError(
            "No barcodes matched keep_labels. Check exact label spelling/case."
        )

    drop_scalar_row_attrs(Sample.dna)
    drop_scalar_row_attrs(Sample.protein)

    dna_sub = copy.copy(Sample.dna)[bars_keep, :]
    protein_sub = copy.copy(Sample.protein)[bars_keep, :]
    Sample_T_subset = MosaicSample(dna=dna_sub, protein=protein_sub)

    vprint(
        f"  Created subset — DNA {Sample_T_subset.dna.shape}, "
        f"Protein {Sample_T_subset.protein.shape}"
    )

    ############################################################################
    # 2. Candidate variants
    ############################################################################

    all_variants = list(
        map(
            str,
            Sample_T_subset.dna.col_attrs.get(
                "id",
                Sample_T_subset.dna.ids(),
            ),
        )
    )

    candidate_variants = set(all_variants)
    n_start = len(candidate_variants)

    ############################################################################
    # 3. Extract NGT, AF, DP
    ############################################################################

    df_ngt = Sample_T_subset.dna.get_attribute("NGT", constraint="row+col")
    df_af = Sample_T_subset.dna.get_attribute("AF", constraint="row+col")

    if not isinstance(df_ngt, pd.DataFrame):
        df_ngt = pd.DataFrame(
            df_ngt,
            index=Sample_T_subset.dna.row_attrs["barcode"],
            columns=Sample_T_subset.dna.col_attrs["id"],
        )

    if not isinstance(df_af, pd.DataFrame):
        df_af = pd.DataFrame(
            df_af,
            index=Sample_T_subset.dna.row_attrs["barcode"],
            columns=Sample_T_subset.dna.col_attrs["id"],
        )

    if min_dp > 0:
        try:
            df_dp = Sample_T_subset.dna.get_attribute("DP", constraint="row+col")

            if not isinstance(df_dp, pd.DataFrame):
                df_dp = pd.DataFrame(
                    df_dp,
                    index=df_ngt.index,
                    columns=df_ngt.columns,
                )

        except Exception as e:
            vprint(f"  ⚠️ DP layer unavailable ({e}); DP filter disabled")
            df_dp = pd.DataFrame(
                np.inf,
                index=df_ngt.index,
                columns=df_ngt.columns,
            )
    else:
        df_dp = pd.DataFrame(
            np.inf,
            index=df_ngt.index,
            columns=df_ngt.columns,
        )

    variant_cols = [v for v in df_ngt.columns if v in candidate_variants]

    df_ngt = df_ngt[variant_cols]
    df_af = df_af[variant_cols]
    df_dp = df_dp[variant_cols]

    vprint(f"\n  Variants in NGT/AF/DP matrices: {len(variant_cols)}")

    ############################################################################
    # 4. Per-variant NGT-first then DP filter
    ############################################################################

    # Mission Bio NGT convention:
    #   0 = WT / hom-ref
    #   1 = HET
    #   2 = HOM-ALT
    #   3 = no genotype / missing
    #
    # First remove NGT == 3 cells per variant.
    called_ngt_mask = df_ngt.isin([0, 1, 2])

    # Then apply DP filter.
    dp_mask = df_dp >= min_dp
    valid_mask = called_ngt_mask & dp_mask

    ngt_valid = df_ngt.where(valid_mask)
    af_valid = df_af.where(valid_mask)

    n_total_cells = df_ngt.shape[0]

    n_called_ngt = called_ngt_mask.sum(axis=0)
    n_valid = valid_mask.sum(axis=0)

    n_ref_valid = ngt_valid.eq(0).sum(axis=0)
    n_het_valid = ngt_valid.eq(1).sum(axis=0)
    n_hom_alt_valid = ngt_valid.eq(2).sum(axis=0)
    n_alt_valid = n_het_valid + n_hom_alt_valid

    n_ngt_missing = df_ngt.eq(3).sum(axis=0)
    n_low_dp_among_called = (called_ngt_mask & ~dp_mask).sum(axis=0)

    ngt_missing_frac = n_ngt_missing / n_total_cells
    call_rate_after_ngt_dp = n_valid / n_total_cells

    ref_frac_valid = n_ref_valid / n_valid.replace(0, np.nan)
    het_frac_valid = n_het_valid / n_valid.replace(0, np.nan)
    hom_alt_frac_valid = n_hom_alt_valid / n_valid.replace(0, np.nan)

    het_frac_among_alt_valid = n_het_valid / n_alt_valid.replace(0, np.nan)

    het_to_hom_alt_ratio = n_het_valid / n_hom_alt_valid.replace(0, np.nan)
    het_to_hom_alt_ratio = het_to_hom_alt_ratio.where(
        n_hom_alt_valid > 0,
        other=np.inf,
    )

    mean_af_valid = af_valid.mean(axis=0)
    median_af_valid = af_valid.median(axis=0)

    ############################################################################
    # 5. Pathogenic exclusion
    ############################################################################

    pathogenic_ids = set()

    if annotation_df is not None:
        vprint(f"\n  Annotation table: {annotation_df.shape[0]} variants")

        anno = annotation_df.copy()

        if annotation_variant_id_col in anno.columns:
            anno["_vid"] = anno[annotation_variant_id_col].astype(str).str.strip()
        else:
            anno["_vid"] = anno.index.astype(str).str.strip()
            vprint("  Using annotation index as variant IDs")

        if exclude_pathogenic and "ClinVar" in anno.columns:
            clinvar = anno["ClinVar"].astype(str).str.lower().fillna("")

            pathogenic_mask = (
                clinvar.str.contains("pathogenic", na=False)
                & ~clinvar.str.contains("benign", na=False)
            )

            pathogenic_ids = set(anno.loc[pathogenic_mask, "_vid"])

            before = len(candidate_variants)
            candidate_variants -= pathogenic_ids

            vprint(
                "  Excluded "
                f"{before - len(candidate_variants)} "
                "Pathogenic/Likely_pathogenic variants"
            )
    else:
        vprint("\n  No annotation_df provided — skipping ClinVar filter")

    ############################################################################
    # 6. Blacklist exclusion
    ############################################################################

    blacklisted_ids = set()

    if blacklist_paths is not None and len(blacklist_paths) > 0:
        parsed = _parse_variant_ids(candidate_variants)
        parsed_valid = parsed.dropna(subset=["chr", "pos"]).copy()
        parsed_valid["pos"] = parsed_valid["pos"].astype(int)

        in_any_blacklist = pd.Series(False, index=parsed_valid.index)

        vprint("\n  Applying blacklist filters")

        for bl_path in blacklist_paths:
            try:
                bl_df = _read_blacklist(bl_path)
                intervals = _build_intervals(bl_df)

                in_bl = parsed_valid.apply(
                    lambda r: _is_blacklisted(r["chr"], r["pos"], intervals),
                    axis=1,
                )

                in_any_blacklist = in_any_blacklist | in_bl

                vprint(
                    f"    {os.path.basename(bl_path)}: "
                    f"{int(in_bl.sum())} variants overlap"
                )

            except Exception as e:
                vprint(f"    ⚠️ Could not read blacklist {bl_path}: {e}")

        blacklisted_ids = set(parsed_valid.loc[in_any_blacklist, "variant"])

        before = len(candidate_variants)
        candidate_variants -= blacklisted_ids

        vprint(f"  Excluded blacklisted variants: {before - len(candidate_variants)}")

    vprint(
        f"\n  Candidates after annotation/blacklist filters: "
        f"{len(candidate_variants)} / {n_start}"
    )

    ############################################################################
    # 7. Load dbSNP/gnomAD common resources and curated HET whitelist
    ############################################################################

    vprint("\n  Loading dbSNP/gnomAD common-variant resources")
    common_variant_ids = _read_id_files(
        common_variant_paths,
        mode=common_requires,
    )

    known_het_ids = set()

    # Direct Python list/set/array, e.g. germline_whitelist.
    if known_het_variants is not None:
        known_het_ids |= _flatten_variant_list(known_het_variants)

    # Optional file-based known-HET anchors.
    if known_het_variant_paths is not None:
        vprint("\n  Loading known/common-HET anchor resources from files")
        known_het_ids |= _read_id_files(
            known_het_variant_paths,
            mode="either",
        )

    # Fallback: if no curated HET anchors are supplied, use dbSNP/gnomAD common IDs.
    if len(known_het_ids) == 0:
        known_het_ids = set(common_variant_ids)

    # Treat curated HET anchors as common so they get the wider 25–75 AF window.
    if treat_known_het_as_common:
        common_variant_ids |= known_het_ids

    vprint(f"  Common-resource IDs: {len(common_variant_ids):,}")
    vprint(f"  Known-HET/whitelist IDs: {len(known_het_ids):,}")

    ############################################################################
    # 8. Build QC table
    ############################################################################

    qc = pd.DataFrame({
        "variant": variant_cols,

        "is_common_resource": [
            v in common_variant_ids for v in variant_cols
        ],
        "is_known_het_whitelist": [
            v in known_het_ids for v in variant_cols
        ],

        "n_total_cells": n_total_cells,
        "n_called_ngt_before_dp": n_called_ngt.reindex(variant_cols).values,
        "n_valid_after_ngt_and_dp": n_valid.reindex(variant_cols).values,
        "n_ngt_missing": n_ngt_missing.reindex(variant_cols).values,
        "n_low_dp_among_called": n_low_dp_among_called.reindex(variant_cols).values,

        "ngt_missing_frac": ngt_missing_frac.reindex(variant_cols).values,
        "call_rate_after_ngt_dp": call_rate_after_ngt_dp.reindex(variant_cols).values,

        "n_ref_valid": n_ref_valid.reindex(variant_cols).values,
        "n_het_valid": n_het_valid.reindex(variant_cols).values,
        "n_hom_alt_valid": n_hom_alt_valid.reindex(variant_cols).values,
        "n_alt_valid": n_alt_valid.reindex(variant_cols).values,

        "ref_frac_valid": ref_frac_valid.reindex(variant_cols).values,
        "het_frac_valid": het_frac_valid.reindex(variant_cols).values,
        "hom_alt_frac_valid": hom_alt_frac_valid.reindex(variant_cols).values,

        "het_to_hom_alt_ratio": het_to_hom_alt_ratio.reindex(variant_cols).values,
        "het_frac_among_alt_valid": het_frac_among_alt_valid.reindex(variant_cols).values,

        "mean_af_valid": mean_af_valid.reindex(variant_cols).values,
        "median_af_valid": median_af_valid.reindex(variant_cols).values,

        "is_pathogenic_excluded": [
            v in pathogenic_ids for v in variant_cols
        ],
        "is_blacklisted": [
            v in blacklisted_ids for v in variant_cols
        ],
    })

    ############################################################################
    # 9. Apply filters
    ############################################################################

    qc["af_min_used"] = np.where(
        qc["is_common_resource"],
        af_min_common,
        af_min_novel,
    )

    qc["af_max_used"] = np.where(
        qc["is_common_resource"],
        af_max_common,
        af_max_novel,
    )

    qc["pass_ngt_dp_basic"] = (
        (qc["n_valid_after_ngt_and_dp"] >= min_valid_cells)
        & (qc["call_rate_after_ngt_dp"] >= min_call_rate_after_ngt_dp)
        & (qc["ngt_missing_frac"] <= max_ngt_frac)
        & (qc["hom_alt_frac_valid"] <= max_hom_frac)
    )

    qc["pass_af_window"] = (
        qc["mean_af_valid"].notna()
        & (qc["mean_af_valid"] >= qc["af_min_used"])
        & (qc["mean_af_valid"] <= qc["af_max_used"])
    )

    if require_het_hom_alt_ratio:
        qc["pass_het_hom_alt_ratio"] = (
            (qc["n_het_valid"] >= min_het_valid_cells)
            & (qc["n_alt_valid"] > 0)
            & (
                (qc["het_to_hom_alt_ratio"] >= min_het_to_hom_alt_ratio)
                | np.isinf(qc["het_to_hom_alt_ratio"])
            )
            & (
                qc["het_frac_among_alt_valid"]
                >= min_het_frac_among_alt_valid
            )
        )
    else:
        qc["pass_het_hom_alt_ratio"] = True

    qc["pass_basic_germline_filter"] = (
        qc["pass_ngt_dp_basic"]
        & qc["pass_af_window"]
        & qc["pass_het_hom_alt_ratio"]
        & ~qc["is_pathogenic_excluded"]
        & ~qc["is_blacklisted"]
    )

    qc["pass_final_germline_filter"] = qc["pass_basic_germline_filter"]

    qc = qc.sort_values(
        [
            "pass_final_germline_filter",
            "is_known_het_whitelist",
            "is_common_resource",
            "het_frac_among_alt_valid",
            "het_to_hom_alt_ratio",
            "call_rate_after_ngt_dp",
            "mean_af_valid",
        ],
        ascending=[False, False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    filtered_germline_variants = (
        qc.loc[qc["pass_final_germline_filter"], "variant"]
        .astype(str)
        .tolist()
    )

    ############################################################################
    # 10. Attach outputs to subset
    ############################################################################

    vprint("\n" + "-" * 60)
    vprint(f"TOTAL GERMLINE VARIANTS: {len(filtered_germline_variants)}")
    vprint("-" * 60)

    if len(filtered_germline_variants) > 0:
        Sample_T_subset.dna_subset = Sample_T_subset.dna[:, filtered_germline_variants]
    else:
        # Mosaic's assay indexing raises IndexError on an empty column list
        # (it inspects key_col[0]). The all-variant profiler uses
        # Sample_T_subset.dna (full panel), not dna_subset, so None is safe.
        Sample_T_subset.dna_subset = None
        vprint("  ⚠️ No variants passed the legacy AF-window filter; "
               "dna_subset set to None.")

    # Useful for downstream inspection.
    Sample_T_subset.germline_qc = qc

    if Sample_T_subset.dna_subset is not None:
        vprint(f"  DNA subset shape: {Sample_T_subset.dna_subset.shape}")

    if verbose and len(filtered_germline_variants) > 0:
        show_cols = [
            "variant",
            "is_common_resource",
            "is_known_het_whitelist",
            "n_valid_after_ngt_and_dp",
            "call_rate_after_ngt_dp",
            "ngt_missing_frac",
            "n_het_valid",
            "n_hom_alt_valid",
            "het_to_hom_alt_ratio",
            "het_frac_among_alt_valid",
            "het_frac_valid",
            "hom_alt_frac_valid",
            "mean_af_valid",
            "median_af_valid",
            "pass_final_germline_filter",
        ]

        show_cols = [c for c in show_cols if c in qc.columns]

        vprint("\n  Top final germline variants:")
        vprint(
            qc.loc[qc["pass_final_germline_filter"], show_cols]
            .head(20)
            .to_string(index=False)
        )

    vprint("\n" + "=" * 60 + "\n")

    if return_qc:
        return filtered_germline_variants, Sample_T_subset, qc

    return filtered_germline_variants, Sample_T_subset


# -----------------------------------------------------------------------------
# AF-shape germline validation helpers
# -----------------------------------------------------------------------------

import re
import numpy as np
import pandas as pd


def _is_indel(variant_id: str) -> bool:
    """True for indels / complex alleles, whose Tapestri AF is not
    interpretable as a diploid 0.5 expectation (and so cannot be AF-validated).
    """
    s = str(variant_id)
    try:
        ref, alt = s.rsplit(":", 1)[1].split("/")
    except Exception:
        return True
    if "+" in alt or "*" in ref or "*" in alt:
        return True
    return len(ref) != 1 or len(alt) != 1


def _cell_by_variant(dna, layer, keep):
    df = dna.get_attribute(layer, constraint="row+col")
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(
            df,
            index=dna.row_attrs["barcode"],
            columns=dna.col_attrs["id"],
        )
    return df[[v for v in df.columns if v in keep]]


def validate_germline_by_af(
    Sample_T_subset,
    candidate_variants,
    common_variant_ids=None,
    known_het_ids=None,
    *,
    # which cells define each variant's AF distribution
    selection_mode: str = "het_calls",   # "het_calls" or "af_band"
    af_band: tuple = (15.0, 95.0),        # used only when selection_mode == "af_band"
    min_dp_af: int = 2,

    # high-confidence HET reference construction
    ref_min_cells: int = 30,
    ref_min_total_dp: int = 300,
    ref_af_lo: float = 40.0,
    ref_af_hi: float = 60.0,
    ref_min_n: int = 10,
    min_ref_spread: float = 3.0,

    # candidate scoring
    k_tol: float = 3.0,
    tol_cap: float = 20.0,
    enable_dispersion: bool = True,
    k_disp: float = 2.5,
    min_eval_total_dp: int = 100,
    min_eval_cells: int = 5,

    # disposition of cases that cannot be AF-validated
    keep_known_low_evidence: bool = True,
    min_weak_cells: int = 3,
    keep_indels: bool = False,

    verbose: bool = True,
):
    """Validate candidate germline HETs by AF-distribution shape.

    NOTE
    ----
    The AF profile is computed only over HET-called cells, so it is blind to a
    dominant WT or HOM-ALT population. A site that is hom-ref or hom-alt in the
    donor, but has a handful of HET-called cells pooling near 50%, can pass
    this AF test. Use the genotype-call structure (see ``build_and_save_all_
    variant_germline_profile`` / ``donor_genotype``) to gate on the donor's
    actual genotype before trusting an AF-validated status.

    Parameters
    ----------
    Sample_T_subset : MosaicSample
        The T-cell subset (must expose NGT / AF / DP layers via
        ``Sample_T_subset.dna``).
    candidate_variants : iterable of str
        Variant IDs surviving pathogenic / blacklist exclusion.
    common_variant_ids, known_het_ids : set, optional
        dbSNP/gnomAD common IDs and the curated HET whitelist. Used to pick
        reference exemplars and to extend benefit of the doubt to known
        germline that is under-sampled here.

    selection_mode : {"het_calls", "af_band"}
        How the cells contributing to a variant's AF distribution are chosen.
        ``"het_calls"`` uses NGT == HET cells; the Mission Bio caller tolerates
        skewed HETs (e.g. reference-biased loci still get HET calls at ~75% AF),
        so genuine skew remains visible. ``"af_band"`` instead takes any
        alt-carrying cell with AF inside ``af_band``, avoiding reliance on the
        genotype caller at the cost of admitting more ref-dropout noise.
    k_tol : float
        Acceptance is ``|pooled_af - centre| <= k_tol * sqrt(ref_spread^2 +
        se_pooled^2)``. The ``se_pooled`` term (= 50 / sqrt(total reads)) is the
        depth-dependent part: shallow variants get a wide tolerance, deep ones
        must sit tightly on the reference centre.

    Returns
    -------
    germline : list[str]
    profile : pd.DataFrame
        Per-variant AF-distribution diagnostics and disposition.
    ref_summary : dict
        Reference centre, spread, and dispersion learned from the sample.
    """

    def vprint(*a):
        if verbose:
            print(*a)

    common_variant_ids = set(common_variant_ids or [])
    known_het_ids = set(known_het_ids or [])
    keep = set(str(v) for v in candidate_variants)

    dna = Sample_T_subset.dna
    df_ngt = _cell_by_variant(dna, "NGT", keep)
    df_af = _cell_by_variant(dna, "AF", keep).astype(float)
    try:
        df_dp = _cell_by_variant(dna, "DP", keep).astype(float)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "DP layer is required for depth-aware AF validation."
        ) from e

    variants = [v for v in df_ngt.columns if v in df_af.columns and v in df_dp.columns]
    df_ngt, df_af, df_dp = df_ngt[variants], df_af[variants], df_dp[variants]

    # ── cells that define each variant's AF distribution ──────────────────
    if selection_mode == "het_calls":
        cell_mask = df_ngt.eq(1) & (df_dp >= min_dp_af)
    elif selection_mode == "af_band":
        lo, hi = af_band
        cell_mask = (
            df_ngt.isin([1, 2]) & (df_dp >= min_dp_af)
            & df_af.ge(lo) & df_af.le(hi)
        )
    else:
        raise ValueError("selection_mode must be 'het_calls' or 'af_band'.")

    af_frac = df_af / 100.0
    m_dp = df_dp.where(cell_mask)
    m_af = df_af.where(cell_mask)
    m_aff = af_frac.where(cell_mask)

    n_cells = cell_mask.sum(axis=0)
    dp_sum = m_dp.sum(axis=0)
    alt_reads = (m_aff * m_dp).sum(axis=0)

    pooled_af = 100.0 * alt_reads / dp_sum.replace(0, np.nan)   # depth-weighted AF
    se_pooled = 50.0 / np.sqrt(dp_sum.replace(0, np.nan))       # binomial SE, percent
    median_af = m_af.median(axis=0)

    # per-cell depth-standardised deviation from 0.5; robust SD flags over-dispersion
    z = 2.0 * np.sqrt(m_dp) * (m_aff - 0.5)
    z_med = z.median(axis=0)
    z_robust_sd = 1.4826 * z.sub(z_med, axis=1).abs().median(axis=0)

    prof = pd.DataFrame({
        "variant": variants,
        "is_common": [v in common_variant_ids for v in variants],
        "is_whitelist": [v in known_het_ids for v in variants],
        "is_indel": [_is_indel(v) for v in variants],
        "n_het_cells": n_cells.reindex(variants).values,
        "total_dp": dp_sum.reindex(variants).values,
        "pooled_af": pooled_af.reindex(variants).values,
        "median_af": median_af.reindex(variants).values,
        "se_pooled_af": se_pooled.reindex(variants).values,
        "z_robust_sd": z_robust_sd.reindex(variants).values,
    })

    # ── 1. high-confidence HET reference (clean, known-germline exemplars) ─
    eligible = prof["is_whitelist"] | prof["is_common"]
    clean = (
        eligible
        & ~prof["is_indel"]
        & (prof["n_het_cells"] >= ref_min_cells)
        & (prof["total_dp"] >= ref_min_total_dp)
        & prof["pooled_af"].between(ref_af_lo, ref_af_hi)
    )
    n_ref = int(clean.sum())

    if n_ref >= ref_min_n:
        centre = float(prof.loc[clean, "pooled_af"].median())
        # SD, not MAD, for the systematic-bias band: mild reference/mapping
        # bias gives genuine germline HETs a depth-independent tail away from
        # 50 (e.g. a deep, low-dispersion locus sitting at 57%). MAD is robust
        # to exactly that tail and so wrongly clips real HETs; the SD admits it.
        # The depth-dependent se term is added separately when scoring.
        ref_spread = max(float(prof.loc[clean, "pooled_af"].std(ddof=1)),
                         min_ref_spread)
        ref_z_sd = float(np.nanmedian(prof.loc[clean, "z_robust_sd"]))
        if not np.isfinite(ref_z_sd) or ref_z_sd <= 0:
            ref_z_sd = 1.0
    else:
        vprint(f"  ⚠️ Only {n_ref} clean reference HETs (< {ref_min_n}); "
               "using centre = 50 and a default spread.")
        centre, ref_spread, ref_z_sd = 50.0, 7.0, 1.5

    vprint("\n  HET reference profile")
    vprint(f"    reference variants : {n_ref}")
    vprint(f"    centre (pooled AF) : {centre:.2f}")
    vprint(f"    assay spread (±)   : {ref_spread:.2f}")
    if enable_dispersion:
        vprint(f"    reference z-SD     : {ref_z_sd:.2f}")

    # ── 2/3. score each candidate's AF distribution against the reference ─
    prof["af_centre"] = centre
    # depth-scaled tolerance, capped: read depth widens the band within a
    # biological sanity bound, but never far enough that a near-homozygous
    # pooled AF (e.g. a single ref-dropout cell read deep) passes as a HET.
    prof["af_tol"] = np.minimum(
        k_tol * np.sqrt(ref_spread**2 + prof["se_pooled_af"] ** 2),
        tol_cap,
    )
    prof["af_deviation"] = (prof["pooled_af"] - centre).abs()

    prof["evidence_ok"] = (
        (prof["total_dp"] >= min_eval_total_dp)
        & (prof["n_het_cells"] >= min_eval_cells)
    )
    prof["centre_ok"] = prof["af_deviation"] <= prof["af_tol"]
    prof["dispersion_ok"] = (
        prof["z_robust_sd"] <= (k_disp * ref_z_sd)
        if enable_dispersion else True
    )

    def _classify(r):
        if r["is_indel"]:
            return "indel_af_unreliable"
        if r["n_het_cells"] == 0:
            return "absent"
        if not r["evidence_ok"]:
            # Curated whitelist gets benefit of the doubt only when its scant
            # evidence is itself HET-consistent (enough cells, centre near 50);
            # dbSNP/gnomAD membership alone is not evidence of presence and
            # never readmits a variant.
            if (r["is_whitelist"]
                    and r["n_het_cells"] >= min_weak_cells
                    and r["centre_ok"]):
                return "whitelist_low_evidence"
            return "insufficient_evidence"
        if not r["centre_ok"]:
            return "af_skewed"
        if not r["dispersion_ok"]:
            return "overdispersed"
        return "validated"

    prof["status"] = prof.apply(_classify, axis=1)

    passed = prof["status"] == "validated"
    if keep_known_low_evidence:
        passed = passed | (prof["status"] == "whitelist_low_evidence")
    if keep_indels:
        passed = passed | (
            (prof["status"] == "indel_af_unreliable")
            & prof["is_whitelist"]
            & (prof["n_het_cells"] >= min_weak_cells)
        )
    prof["pass_germline"] = passed

    prof = prof.sort_values(
        ["pass_germline", "status", "af_deviation"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    germline = prof.loc[prof["pass_germline"], "variant"].tolist()

    if verbose:
        vprint("\n  Disposition:")
        vprint(prof["status"].value_counts().to_string())
        vprint(f"\n  Validated germline HETs: {len(germline)}")

    return germline, prof, {
        "n_reference": n_ref,
        "centre": centre,
        "spread": ref_spread,
        "z_sd": ref_z_sd,
    }


def review_failed_variants(
    prof,
    *,
    ref_z_sd: float,
    review_status=("af_skewed", "insufficient_evidence"),
    min_review_cells: int = 30,
    consistency_mult: float = 1.0,
    ref_like_below: float = 15.0,
    hom_like_above: float = 82.0,
    near_window: int = 50_000,
    verbose: bool = True,
):
    """Second-pass triage of AF-failed variants.

    A variant that fails AF-centre validation is not necessarily unusable: a
    germline HET on a biased amplicon reads at a reproducibly skewed AF (e.g.
    25% or 72%) yet is still reliably HET-genotyped, and genotype-based
    demultiplexing can use it. This pass separates those from genuine
    homozygous sites, near-reference noise, and likely mapping artefacts.

    Flags, surfaced as columns for human judgement rather than baked into a
    single verdict:

      consistent      : tight per-cell AF (z_robust_sd <= consistency_mult ×
                        reference z-SD) across >= min_review_cells cells — the
                        skew is systematic, not noise.
      af_zone         : ref_like (<ref_like_below) / biased_het_low /
                        biased_het_high / hom_like (>hom_like_above).
      chrX_nonPAR     : chrX outside PAR1/PAR2 (hg19) — a diploid-style HET is
                        impossible here in a male donor; scrutinise.
      n_near_failed   : other failed variants within ``near_window`` bp on the
                        same chromosome — a tight pack can indicate a region-
                        level CNV / mismapping (weaker signal in a targeted
                        panel, where amplicons legitimately cluster on genes).

    verdict
      likely_usable_het : consistent, many cells, AF skewed but not at a
                          homozygous/reference extreme — candidate to add back.
      review_hom_or_ref : AF sits at a homozygous or near-reference value;
                          probably not a HET (may still be a valid germline
                          marker, just not heterozygous).
      inconclusive      : dispersed or too few cells to judge.
    """
    df = prof[prof["status"].isin(review_status) & ~prof["is_indel"]].copy()
    if df.empty:
        if verbose:
            print("  No reviewable variants.")
        return df

    coords = df["variant"].str.extract(r"^(chr[^:]+):(\d+):")
    df["chrom"] = coords[0]
    df["pos"] = pd.to_numeric(coords[1], errors="coerce")

    df["n_near_failed"] = [
        int((
            (df["chrom"] == r.chrom)
            & (df["variant"] != r.variant)
            & (df["pos"].sub(r.pos).abs() <= near_window)
        ).sum())
        for r in df.itertuples()
    ]

    def _par(p):
        return (60001 <= p <= 2699520) or (154931044 <= p <= 155260560)

    df["chrX_nonPAR"] = (df["chrom"] == "chrX") & ~df["pos"].map(
        lambda p: _par(p) if pd.notna(p) else True
    )

    df["consistent"] = (
        (df["z_robust_sd"] <= consistency_mult * ref_z_sd)
        & (df["n_het_cells"] >= min_review_cells)
    )

    def _zone(af):
        if pd.isna(af):
            return "na"
        if af < ref_like_below:
            return "ref_like"
        if af > hom_like_above:
            return "hom_like"
        return "biased_het_low" if af < 50 else "biased_het_high"

    df["af_zone"] = df["pooled_af"].map(_zone)

    def _verdict(r):
        if r["af_zone"] in ("hom_like", "ref_like"):
            return "review_hom_or_ref"
        if not r["consistent"]:
            return "inconclusive"
        return "likely_usable_het"

    df["verdict"] = df.apply(_verdict, axis=1)

    order = {"likely_usable_het": 0, "review_hom_or_ref": 1, "inconclusive": 2}
    df["_o"] = df["verdict"].map(order).fillna(3)
    df = (df.sort_values(["_o", "n_het_cells"], ascending=[True, False])
            .drop(columns="_o").reset_index(drop=True))

    if verbose:
        print("  Second-pass review of AF-failed variants")
        print(df["verdict"].value_counts().to_string())
        print()
        cols = ["variant", "is_common", "is_whitelist", "n_het_cells",
                "total_dp", "pooled_af", "z_robust_sd", "af_zone",
                "n_near_failed", "chrX_nonPAR", "verdict"]
        print(df[cols].head(40).to_string(index=False))

    return df


def genotype_af_cutoffs(
    Sample_T_subset,
    *,
    min_dp: int = 5,
    af_bin: float = 2.0,
    min_bin_n: int = 50,
    dp_bands=((5, 15), (15, 40), (40, np.inf)),
    evaluate_variants=None,
    plot: bool = True,
    ax=None,
):
    """Empirically locate the AF thresholds at which the genotype caller
    switches cells out of HET — into WT below and HOM-ALT above — by pooling
    every cell's (AF, NGT) across all variants.

    The crossover where HET overtakes WT is the lower cutoff; where HOM-ALT
    overtakes HET is the upper cutoff. Between them, HET is the caller's
    dominant decision: that is the AF range in which a heterozygous genotype
    is credible at this depth. The cutoffs widen at low depth, so they are
    also reported per DP band.

    With ``evaluate_variants`` set, returns a per-variant placement table:
    the WT/HET/HOM-ALT call structure and where the HET-cell AF sits relative
    to the band. A reproducibly biased true HET shows HET-dominant calls with
    a WT (dropout) tail, ~no HOM-ALT, and a HET-AF distribution sitting wholly
    above the lower cutoff; a homozygous site shows a HOM-ALT mass with the
    HET calls as its low tail.

    Returns
    -------
    cutoffs : dict           {"lower", "upper"} for the pooled data
    binned  : pd.DataFrame    per-AF-bin call fractions (for plotting/inspection)
    bands   : pd.DataFrame    cutoffs per DP band
    placement : pd.DataFrame | None
    """
    dna = Sample_T_subset.dna

    def _full(layer):
        d = dna.get_attribute(layer, constraint="row+col")
        if not isinstance(d, pd.DataFrame):
            d = pd.DataFrame(d, index=dna.row_attrs["barcode"],
                             columns=dna.col_attrs["id"])
        return d

    df_ngt = _full("NGT")
    df_af = _full("AF").astype(float)
    df_dp = _full("DP").astype(float)

    ngt = df_ngt.values.ravel()
    af = df_af.values.ravel()
    dp = df_dp.values.ravel()

    edges = np.arange(0.0, 100.0 + af_bin, af_bin)
    centres = (edges[:-1] + edges[1:]) / 2.0

    def _cutoffs(mask):
        a, g = af[mask], ngt[mask]
        if a.size == 0:
            return np.nan, np.nan, None
        b = np.clip(np.digitize(a, edges) - 1, 0, len(centres) - 1)
        tab = pd.DataFrame({"b": b, "g": g})
        grp = tab.groupby("b")["g"]
        n = grp.size().reindex(range(len(centres)))
        fwt = grp.apply(lambda x: (x == 0).mean()).reindex(range(len(centres)))
        fhet = grp.apply(lambda x: (x == 1).mean()).reindex(range(len(centres)))
        fhom = grp.apply(lambda x: (x == 2).mean()).reindex(range(len(centres)))

        ok = (n >= min_bin_n).fillna(False).values
        c = centres[ok]
        lo = c[(fhet.values[ok] > fwt.values[ok])]
        hi = c[(fhet.values[ok] > fhom.values[ok])]
        lower = float(lo.min()) if lo.size else np.nan
        upper = float(hi.max()) if hi.size else np.nan

        binned = pd.DataFrame({
            "af_centre": centres, "n": n.values,
            "frac_WT": fwt.values, "frac_HET": fhet.values,
            "frac_HOM": fhom.values,
        })
        return lower, upper, binned

    base_mask = np.isin(ngt, [0, 1, 2]) & (dp >= min_dp) & np.isfinite(af)
    lower, upper, binned = _cutoffs(base_mask)

    band_rows = []
    for lo_dp, hi_dp in dp_bands:
        m = np.isin(ngt, [0, 1, 2]) & (dp >= lo_dp) & (dp < hi_dp) & np.isfinite(af)
        bl, bu, _ = _cutoffs(m)
        band_rows.append({"dp_lo": lo_dp, "dp_hi": hi_dp,
                          "n_cells": int(m.sum()), "lower": bl, "upper": bu})
    bands = pd.DataFrame(band_rows)

    placement = None
    if evaluate_variants is not None:
        placement = compute_genotype_structure(
            Sample_T_subset, evaluate_variants,
            min_dp=min_dp, lower=lower, upper=upper,
        )

    if plot:
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(binned["af_centre"], binned["frac_WT"], label="WT", lw=1.8)
        ax.plot(binned["af_centre"], binned["frac_HET"], label="HET", lw=1.8)
        ax.plot(binned["af_centre"], binned["frac_HOM"], label="HOM-ALT", lw=1.8)
        for x, lbl in [(lower, "lower"), (upper, "upper")]:
            if np.isfinite(x):
                ax.axvline(x, color="0.4", ls="--", lw=1)
                ax.text(x, 1.02, f"{lbl}\n{x:.0f}%", ha="center", va="bottom",
                        fontsize=8, color="0.3")
        ax.set_xlabel("Per-cell AF (%)")
        ax.set_ylabel("Fraction of cells with call")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"Genotype call vs AF (DP \u2265 {min_dp})", fontsize=10)

    return {"lower": lower, "upper": upper}, binned, bands, placement


# -----------------------------------------------------------------------------
# All-variant germline profiling helpers
# -----------------------------------------------------------------------------

import os
import glob
import re
import numpy as np
import pandas as pd


GERMLINE_PATTERN_DIR = (
    "/project/meadlab/shared/02__Projects/02b__Tapestri/"
    "resources/variants_patterns_tapestri"
)


def _dedupe_keep_order(items):
    """Return unique stringified values, preserving input order."""
    seen = set()
    out = []

    for x in items:
        if pd.isna(x):
            continue
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out


def _normalise_variant_list(x):
    """
    Normalise whitelist/common/resource inputs to a list of variant strings.
    Accepts list/set/tuple/Index/Series/DataFrame.
    """
    if x is None:
        return []

    if isinstance(x, pd.DataFrame):
        for col in ["variant", "id", "ID", "Variant"]:
            if col in x.columns:
                return _dedupe_keep_order(x[col].tolist())
        return _dedupe_keep_order(x.iloc[:, 0].tolist())

    if isinstance(x, pd.Series):
        return _dedupe_keep_order(x.tolist())

    if isinstance(x, pd.Index):
        return _dedupe_keep_order(x.tolist())

    return _dedupe_keep_order(list(x))


def _coerce_bool_series(s):
    """
    Robust bool conversion for columns that may contain bools, 0/1, strings,
    missing values, etc.
    """
    if s is None:
        return None

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).ne(0)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def _full_layer_df(dna, layer):
    """
    Return a cell × variant dataframe for a Tapestri DNA layer.
    """
    df = dna.get_attribute(layer, constraint="row+col")

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(
            df,
            index=np.asarray(dna.row_attrs["barcode"]).astype(str),
            columns=np.asarray(dna.col_attrs["id"]).astype(str),
        )

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    return df


def compute_genotype_structure(
    Sample_T_subset,
    variants,
    *,
    min_dp: int = 5,
    lower=np.nan,
    upper=np.nan,
):
    """Per-variant WT/HET/HOM-ALT call structure over cells with DP >= min_dp.

    This is the donor-genotype evidence: it sees the whole called population,
    not just HET-called cells, so it distinguishes a heterozygous donor (HET
    present with a WT dropout tail and ~no HOM-ALT) from a homozygous-ref site
    (WT-dominated, sporadic HET noise) and a homozygous-alt site (HOM-dominated).
    """
    dna = Sample_T_subset.dna
    df_ngt = _full_layer_df(dna, "NGT")
    df_af = _full_layer_df(dna, "AF").astype(float)
    df_dp = _full_layer_df(dna, "DP").astype(float)

    rows = []
    for v in _dedupe_keep_order(variants):
        if v not in df_ngt.columns:
            continue
        mm = df_ngt[v].isin([0, 1, 2]) & (df_dp[v] >= min_dp)
        gg = df_ngt[v][mm]
        aa = df_af[v][mm]
        het = aa[gg == 1]
        n = int(mm.sum())
        rows.append({
            "variant": v,
            "n_cells": n,
            "frac_WT": float((gg == 0).mean()) if n else np.nan,
            "frac_HET": float((gg == 1).mean()) if n else np.nan,
            "frac_HOM": float((gg == 2).mean()) if n else np.nan,
            "het_af_p10": float(np.percentile(het, 10)) if het.size else np.nan,
            "het_af_median": float(het.median()) if het.size else np.nan,
            "het_af_p90": float(np.percentile(het, 90)) if het.size else np.nan,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["het_pop_above_lower"] = out["het_af_p10"] >= lower
        out["hom_dominated"] = out["frac_HOM"] > out["frac_HET"]
        out["margin_to_lower"] = out["het_af_p10"] - lower
        out["margin_to_upper"] = upper - out["het_af_p90"]

    return out


def _donor_genotype_call(fwt, fhet, fhom, n_cells, *, min_het, max_hom, max_wt,
                         min_cells=1):
    """Infer the donor genotype from the cell-level call structure, independent
    of AF shape.

    A germline HET requires the donor to genotype heterozygous in a majority of
    confidently-called cells (``fhet >= min_het``), with enough cells to trust
    the call (``n_cells >= min_cells``). It is disqualified if HOM-ALT-dominated
    (``fhom >= max_hom``) or WT-dominated (``fwt >= max_wt``). Because WT/HET/HOM
    partition the called cells, ``fwt`` and ``fhet`` are anti-correlated, so
    ``min_het`` is effectively the anti-WT-dominance knob: a site that is mostly
    WT cannot be told apart from a hom-ref site with sporadic noise HET calls,
    so it is not a credible germline HET.

    Returns one of: "het", "hom_alt", "hom_ref", "low_occupancy",
    "low_coverage", "uncalled".
    """
    n = n_cells or 0
    if fhet is None or pd.isna(fhet) or n == 0:
        return "uncalled"
    if n < min_cells:
        return "low_coverage"
    if fhom is not None and not pd.isna(fhom) and fhom >= max_hom:
        return "hom_alt"
    if fwt is not None and not pd.isna(fwt) and fwt >= max_wt:
        return "hom_ref"
    if fhet >= min_het:
        return "het"
    return "low_occupancy"


def variant_level_ngt_af_dp_profile(
    Sample_T_subset,
    variants,
    *,
    min_dp: int = 2,
):
    """
    Build broad per-variant NGT / AF / DP summaries.

    This is deliberately descriptive rather than decisive. It is intended for
    cross-patient threshold discovery.
    """
    dna = Sample_T_subset.dna

    df_ngt = _full_layer_df(dna, "NGT")
    df_af = _full_layer_df(dna, "AF").astype(float)
    df_dp = _full_layer_df(dna, "DP").astype(float)

    variants = _dedupe_keep_order(variants)
    present = set(
        v for v in variants
        if v in df_ngt.columns and v in df_af.columns and v in df_dp.columns
    )

    rows = []

    for v in variants:
        if v not in present:
            rows.append({
                "variant": v,
                "present_in_T_subset": False,
            })
            continue

        ngt = pd.to_numeric(df_ngt[v], errors="coerce")
        af = pd.to_numeric(df_af[v], errors="coerce")
        dp = pd.to_numeric(df_dp[v], errors="coerce")

        valid_dp = dp >= min_dp
        valid_call = ngt.isin([0, 1, 2]) & valid_dp & np.isfinite(af)

        wt = valid_call & ngt.eq(0)
        het = valid_call & ngt.eq(1)
        hom = valid_call & ngt.eq(2)

        alt_like = valid_call & ngt.isin([1, 2])

        af_valid = af[valid_call]
        af_het = af[het]
        af_alt_like = af[alt_like]
        dp_valid = dp[valid_call]

        n_valid = int(valid_call.sum())

        total_dp_valid = float(dp_valid.sum()) if n_valid else np.nan
        alt_reads_valid = (
            float(((af[valid_call] / 100.0) * dp[valid_call]).sum())
            if n_valid
            else np.nan
        )

        pooled_af_all_valid = (
            100.0 * alt_reads_valid / total_dp_valid
            if np.isfinite(total_dp_valid) and total_dp_valid > 0
            else np.nan
        )

        rows.append({
            "variant": v,
            "present_in_T_subset": True,

            "n_T_cells_total": int(len(ngt)),
            "n_valid_dp_ge_min": int(valid_dp.sum()),
            "n_valid_called": n_valid,

            "n_WT": int(wt.sum()),
            "n_HET": int(het.sum()),
            "n_HOM_ALT": int(hom.sum()),
            "n_invalid_or_lowDP": int((~valid_call).sum()),

            "frac_WT": float(wt.sum() / n_valid) if n_valid else np.nan,
            "frac_HET": float(het.sum() / n_valid) if n_valid else np.nan,
            "frac_HOM_ALT": float(hom.sum() / n_valid) if n_valid else np.nan,

            "median_DP_valid": float(dp_valid.median()) if n_valid else np.nan,
            "mean_DP_valid": float(dp_valid.mean()) if n_valid else np.nan,
            "total_DP_valid": total_dp_valid,
            "alt_reads_valid": alt_reads_valid,
            "pooled_AF_all_valid": pooled_af_all_valid,

            "AF_all_valid_p10": float(np.percentile(af_valid, 10)) if len(af_valid) else np.nan,
            "AF_all_valid_median": float(np.percentile(af_valid, 50)) if len(af_valid) else np.nan,
            "AF_all_valid_p90": float(np.percentile(af_valid, 90)) if len(af_valid) else np.nan,

            "AF_HET_p10": float(np.percentile(af_het, 10)) if len(af_het) else np.nan,
            "AF_HET_median": float(np.percentile(af_het, 50)) if len(af_het) else np.nan,
            "AF_HET_p90": float(np.percentile(af_het, 90)) if len(af_het) else np.nan,

            "AF_alt_like_p10": float(np.percentile(af_alt_like, 10)) if len(af_alt_like) else np.nan,
            "AF_alt_like_median": float(np.percentile(af_alt_like, 50)) if len(af_alt_like) else np.nan,
            "AF_alt_like_p90": float(np.percentile(af_alt_like, 90)) if len(af_alt_like) else np.nan,
        })

    return pd.DataFrame(rows)


def build_and_save_all_variant_germline_profile(
    Sample,
    DNA_Utils=None,
    *,
    patient_id: str,
    annotation_df,
    known_het_variants=None,
    keep_labels=("CD4_T", "CD8_T", "Other_T"),
    out_dir: str = GERMLINE_PATTERN_DIR,
    descriptive_min_dp: int = 2,
    placement_min_dp: int = 5,
    # donor-genotype structure gate (the primary cross-patient knob)
    min_frac_het_for_het: float = 0.55,
    max_frac_hom_for_het: float = 0.20,
    max_frac_wt_for_het: float = 0.90,
    min_called_cells_for_genotype: int = 20,
    min_cells_for_run: int = 50,
    verbose: bool = True,
):
    """
    Save one CSV containing all available per-variant germline evidence.

    One row per variant.

    Includes:
      - original DNA_Utils.find_germline_variants QC flags
      - AF-shape validation metrics
      - NGT / AF / DP descriptive metrics
      - empirical genotype-caller AF placement metrics
      - donor genotype inferred from the call structure (``donor_genotype``)
      - second-pass review / rescue verdict
      - broad seed labels for cross-patient threshold learning

    Classification
    --------------
    A germline HET (clean or rescued) requires the DONOR to genotype as
    heterozygous, decided from the WT/HET/HOM-ALT call structure
    (``donor_genotype``) *before* AF shape is considered. The AF profile is
    computed only over HET-called cells and is therefore blind to a dominant
    WT or HOM-ALT population; the structure gate corrects for that, so a
    93%-WT (hom-ref) or 99%-HOM (hom-alt) site can never be seeded even if its
    handful of HET-called cells pool near 50%.

    ``min_frac_het_for_het`` is the primary cross-patient knob. It is set
    conservatively by default; lower-occupancy candidates are retained in the
    output under ``het_low_occupancy_review`` (not seeded) so the cohort can
    decide the floor empirically. No variant is dropped from the CSV.
    """

    os.makedirs(out_dir, exist_ok=True)

    known_het_variants = _normalise_variant_list(known_het_variants)

    # Initial QC table from the germline discovery function.
    # qc remains the master table, so excluded variants are retained.
    # When this helper lives inside DNA_Utils/germline.py, DNA_Utils may be omitted.
    finder = find_germline_variants if DNA_Utils is None else DNA_Utils.find_germline_variants

    _, Sample_T_subset, qc = finder(
        Sample,
        keep_labels=list(keep_labels),
        annotation_df=annotation_df,
        known_het_variants=known_het_variants,
        return_qc=True,
        verbose=False,
    )

    if qc is None or qc.empty:
        raise RuntimeError("find_germline_variants returned an empty qc dataframe.")

    if "variant" not in qc.columns:
        raise RuntimeError("qc dataframe does not contain a 'variant' column.")

    qc = qc.copy()
    qc["variant"] = qc["variant"].astype(str)

    n_qc_before = qc.shape[0]
    qc = qc.drop_duplicates("variant", keep="first").reset_index(drop=True)

    if verbose and qc.shape[0] != n_qc_before:
        print(f"  ⚠ Dropped {n_qc_before - qc.shape[0]} duplicate qc rows by variant ID.")

    all_variants = qc["variant"].tolist()

    # Cell-count guard: germline calling needs a real T-cell population.
    n_t_cells = (
        int(qc["n_total_cells"].iloc[0])
        if "n_total_cells" in qc.columns and len(qc)
        else int(getattr(Sample_T_subset.dna, "shape", (0,))[0])
    )
    run_underpowered = n_t_cells < min_cells_for_run

    if run_underpowered:
        print(
            f"  ⚠️ UNDER-POWERED: only {n_t_cells} cells in the T-cell subset "
            f"(keep_labels={list(keep_labels)}), below min_cells_for_run="
            f"{min_cells_for_run}.\n"
            "     Germline calling is not statistically meaningful here — every "
            "variant will fall to low_coverage / insufficient and no seeds will "
            "be produced.\n"
            "     Check the protein cell-type labels for this sample "
            "(Sample.protein.get_labels()): the labels may be missing, named "
            "differently, or this may simply be a low-input sample.\n"
            "     The CSV is still written (flagged run_underpowered=True) for "
            "record-keeping."
        )

    # Ensure required boolean flags exist and are real bools.
    for col in [
        "is_pathogenic_excluded",
        "is_blacklisted",
        "is_common_resource",
        "is_known_het_whitelist",
    ]:
        if col not in qc.columns:
            qc[col] = False
        qc[col] = _coerce_bool_series(qc[col])

    not_excluded = (
        ~qc["is_pathogenic_excluded"]
        & ~qc["is_blacklisted"]
    )

    # These sets are used by the AF validator as trusted/reference candidates.
    # Excluded/blacklisted variants remain in the output dataframe but are not
    # allowed to drive the reference profile or low-evidence rescue.
    common_for_reference = set(
        qc.loc[not_excluded & qc["is_common_resource"], "variant"]
    )

    whitelist_for_reference = set(
        qc.loc[not_excluded & qc["is_known_het_whitelist"], "variant"]
    )

    # Validate ALL variants, not only the current candidate set.
    germline, prof, ref = validate_germline_by_af(
        Sample_T_subset,
        all_variants,
        common_variant_ids=common_for_reference,
        known_het_ids=whitelist_for_reference,
        verbose=False,
    )

    prof = prof.copy()
    prof["variant"] = prof["variant"].astype(str)

    # Second-pass review of AF-failed / insufficient-evidence variants.
    review = review_failed_variants(
        prof,
        ref_z_sd=ref["z_sd"],
        verbose=False,
    )

    if review is not None and not review.empty:
        review_cols = [
            c for c in [
                "variant",
                "chrom",
                "pos",
                "n_near_failed",
                "chrX_nonPAR",
                "consistent",
                "af_zone",
                "verdict",
            ]
            if c in review.columns
        ]

        review_small = review[review_cols].copy()
        review_small["variant"] = review_small["variant"].astype(str)

        review_small = review_small.rename(columns={
            "chrom": "review_chrom",
            "pos": "review_pos",
            "n_near_failed": "review_n_near_failed",
            "chrX_nonPAR": "review_chrX_nonPAR",
            "consistent": "review_consistent",
            "af_zone": "review_af_zone",
            "verdict": "review_verdict",
        })
    else:
        review_small = pd.DataFrame({"variant": []})

    # Descriptive NGT / AF / DP summaries for every variant.
    descriptive = variant_level_ngt_af_dp_profile(
        Sample_T_subset,
        all_variants,
        min_dp=descriptive_min_dp,
    )

    # Empirical genotype-caller AF placement for every variant.
    cutoffs, binned, bands, placement = genotype_af_cutoffs(
        Sample_T_subset,
        min_dp=placement_min_dp,
        evaluate_variants=all_variants,
        plot=False,
    )

    if placement is not None and not placement.empty:
        placement = placement.copy()
        placement["variant"] = placement["variant"].astype(str)

        placement = placement.rename(columns={
            "n_cells": "placement_n_cells",
            "frac_WT": "placement_frac_WT",
            "frac_HET": "placement_frac_HET",
            "frac_HOM": "placement_frac_HOM",
            "het_af_p10": "placement_HET_AF_p10",
            "het_af_median": "placement_HET_AF_median",
            "het_af_p90": "placement_HET_AF_p90",
            "het_pop_above_lower": "placement_HET_pop_above_lower",
            "hom_dominated": "placement_HOM_dominated",
            "margin_to_lower": "placement_margin_to_lower",
            "margin_to_upper": "placement_margin_to_upper",
        })
    else:
        placement = pd.DataFrame({"variant": []})

    # Rename validator reference-membership columns to avoid confusion with
    # original qc resource flags.
    prof_for_merge = prof.rename(columns={
        "is_common": "afval_is_common_reference",
        "is_whitelist": "afval_is_whitelist_reference",
    })

    # Merge everything onto qc so every original qc variant is retained.
    all_info = (
        qc
        .merge(prof_for_merge, on="variant", how="left", suffixes=("", "_af_validation"))
        .merge(descriptive, on="variant", how="left")
        .merge(review_small, on="variant", how="left")
        .merge(placement, on="variant", how="left")
    )

    # Attach patient/run-level metadata.
    all_info.insert(0, "patient_id", patient_id)

    all_info["is_common_for_reference"] = all_info["variant"].isin(common_for_reference)
    all_info["is_whitelist_for_reference"] = all_info["variant"].isin(whitelist_for_reference)

    all_info["ref_n_reference"] = ref.get("n_reference", np.nan)
    all_info["ref_centre"] = ref.get("centre", np.nan)
    all_info["ref_spread"] = ref.get("spread", np.nan)
    all_info["ref_z_sd"] = ref.get("z_sd", np.nan)

    all_info["caller_AF_lower_cutoff"] = cutoffs.get("lower", np.nan)
    all_info["caller_AF_upper_cutoff"] = cutoffs.get("upper", np.nan)

    all_info["descriptive_min_dp"] = descriptive_min_dp
    all_info["placement_min_dp"] = placement_min_dp
    all_info["min_frac_het_for_het"] = min_frac_het_for_het
    all_info["max_frac_hom_for_het"] = max_frac_hom_for_het
    all_info["max_frac_wt_for_het"] = max_frac_wt_for_het
    all_info["min_called_cells_for_genotype"] = min_called_cells_for_genotype
    all_info["n_T_cells_in_subset"] = n_t_cells
    all_info["run_underpowered"] = run_underpowered

    # ── Donor genotype from the call structure (independent of AF shape) ──
    # Prefer the DP>=placement_min_dp structure; fall back to the descriptive
    # DP>=descriptive_min_dp structure when placement is unavailable.
    def _struct_fracs(r):
        fwt = r.get("placement_frac_WT")
        fhet = r.get("placement_frac_HET")
        fhom = r.get("placement_frac_HOM")
        ncel = r.get("placement_n_cells")
        if fhet is None or pd.isna(fhet):
            fwt = r.get("frac_WT")
            fhet = r.get("frac_HET")
            fhom = r.get("frac_HOM_ALT")
            ncel = r.get("n_valid_called")
        return fwt, fhet, fhom, ncel

    all_info["donor_genotype"] = all_info.apply(
        lambda r: _donor_genotype_call(
            *_struct_fracs(r),
            min_het=min_frac_het_for_het,
            max_hom=max_frac_hom_for_het,
            max_wt=max_frac_wt_for_het,
            min_cells=min_called_cells_for_genotype,
        ),
        axis=1,
    )

    # ── Broad classification for cross-patient threshold discovery ──
    # The donor genotype gates whether a variant can be a germline HET at all;
    # AF shape (status / review verdict) then decides clean vs rescued.
    def _broad_class(r):
        if bool(r.get("is_pathogenic_excluded", False)):
            return "excluded_pathogenic"
        if bool(r.get("is_blacklisted", False)):
            return "excluded_blacklisted"

        status = r.get("status", np.nan)
        if pd.isna(status):
            return "missing_from_T_subset_or_unprofiled"
        if status == "indel_af_unreliable":
            return "indel_af_unreliable"

        dg = r.get("donor_genotype", "uncalled")
        if dg == "uncalled":
            return "absent"
        if dg == "low_coverage":
            return "low_coverage_review"
        if dg == "hom_alt":
            return "likely_hom_alt"
        if dg == "hom_ref":
            return "likely_hom_ref"
        if dg == "low_occupancy":
            return "het_low_occupancy_review"

        # Donor is heterozygous -> AF shape decides clean vs rescued.
        review_verdict = r.get("review_verdict", np.nan)
        if status == "validated":
            return "clean_germline_het"
        if status == "whitelist_low_evidence":
            return "rescued_germline_het_whitelist_low_evidence"
        if review_verdict == "likely_usable_het":
            return "rescued_germline_het_candidate"
        if review_verdict == "review_hom_or_ref":
            return "likely_hom_or_ref"
        if status == "insufficient_evidence":
            return "insufficient_evidence"
        if status == "af_skewed":
            return "af_skewed_not_rescued"
        if status == "overdispersed":
            return "overdispersed"
        if status == "absent":
            return "absent"
        return str(status)

    all_info["germline_broad_class"] = all_info.apply(_broad_class, axis=1)

    all_info["clean_germline_het_seed"] = (
        all_info["germline_broad_class"] == "clean_germline_het"
    )

    all_info["rescued_germline_het_seed"] = all_info["germline_broad_class"].isin([
        "rescued_germline_het_whitelist_low_evidence",
        "rescued_germline_het_candidate",
    ])

    all_info["usable_germline_het_seed"] = (
        all_info["clean_germline_het_seed"]
        | all_info["rescued_germline_het_seed"]
    )

    # Useful sorting: likely useful germline markers first.
    sort_cols = [
        "usable_germline_het_seed",
        "clean_germline_het_seed",
        "rescued_germline_het_seed",
        "germline_broad_class",
        "af_deviation",
    ]
    sort_cols = [c for c in sort_cols if c in all_info.columns]

    all_info = all_info.sort_values(
        sort_cols,
        ascending=[False, False, False, True, True][:len(sort_cols)],
    ).reset_index(drop=True)

    out_csv = os.path.join(
        out_dir,
        f"{patient_id}_variants_germline_data.csv",
    )

    all_info.to_csv(out_csv, index=False)

    if verbose:
        print("Saved all-variant germline profile:")
        print(f"  {out_csv}")
        print()
        print(f"Rows saved: {all_info.shape[0]}")
        print()
        print("Donor genotype counts:")
        print(all_info["donor_genotype"].value_counts(dropna=False).to_string())
        print()
        print("Broad class counts:")
        print(all_info["germline_broad_class"].value_counts(dropna=False).to_string())
        print()
        print(f"Clean germline HET seed:   {int(all_info['clean_germline_het_seed'].sum())}")
        print(f"Rescued germline HET seed: {int(all_info['rescued_germline_het_seed'].sum())}")
        print(f"Usable germline HET seed:  {int(all_info['usable_germline_het_seed'].sum())}")

    return all_info, Sample_T_subset, {
        "germline": germline,
        "ref": ref,
        "cutoffs": cutoffs,
        "binned": binned,
        "bands": bands,
        "prof": prof,
        "review": review,
        "out_csv": out_csv,
        "n_T_cells": n_t_cells,
        "run_underpowered": run_underpowered,
    }


# -----------------------------------------------------------------------------
# Notebook-facing all-variant germline profiling wrapper
# -----------------------------------------------------------------------------

def _safe_patient_export_id(experiment, patient_id):
    """Return a filesystem-safe export identifier from experiment and patient ID."""
    import re

    export_id = f"{experiment}_{patient_id}"
    export_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(export_id))
    export_id = re.sub(r"_+", "_", export_id).strip("_")
    return export_id


def run_all_variant_germline_profile_export(
    Sample,
    *,
    experiment: str,
    patient_id: str,
    annotation_df,
    known_het_variants=None,
    DNA_Utils=None,
    keep_labels=("CD4_T", "CD8_T", "Other_T"),
    out_dir: str = GERMLINE_PATTERN_DIR,
    descriptive_min_dp: int = 2,
    placement_min_dp: int = 5,
    # donor-genotype structure gate (the primary cross-patient knob)
    min_frac_het_for_het: float = 0.55,
    max_frac_hom_for_het: float = 0.20,
    max_frac_wt_for_het: float = 0.90,
    min_called_cells_for_genotype: int = 20,
    min_cells_for_run: int = 50,
    verbose: bool = True,
):
    """Run the all-variant germline profiling export for one patient.

    This is a notebook-friendly wrapper around
    :func:`build_and_save_all_variant_germline_profile`. It creates a safe
    ``<experiment>_<patient_id>`` export ID, writes the all-variant CSV, and
    adds convenience handles to the returned metadata dictionary:

      - ``germline``: usable germline HET seed variants
      - ``clean_germline``: clean germline HET seed variants
      - ``rescued_germline``: rescued germline HET seed variants
      - ``ref``: AF-shape reference profile
      - ``patient_id_for_export``: filesystem-safe export ID

    Parameters
    ----------
    Sample
        Mission Bio Mosaic sample object.
    experiment, patient_id
        Values used to create a safe export ID and CSV prefix.
    annotation_df
        Variant annotation dataframe passed to germline discovery.
    known_het_variants
        Optional curated germline/known-HET variant list.
    DNA_Utils
        Optional module/object exposing ``find_germline_variants``. Leave as
        ``None`` when calling this function from inside the same DNA_Utils
        module.
    keep_labels
        Protein labels used to define the normal/T-cell subset.
    out_dir
        Directory for the exported all-variant germline profile CSV.
    descriptive_min_dp, placement_min_dp
        DP thresholds used for descriptive and genotype-placement summaries.
    min_frac_het_for_het, max_frac_hom_for_het, max_frac_wt_for_het
        Donor-genotype structure thresholds. A germline HET requires HET
        occupancy >= ``min_frac_het_for_het`` and HOM-ALT occupancy
        < ``max_frac_hom_for_het``; WT occupancy >= ``max_frac_wt_for_het`` with
        too little HET is called hom-ref. ``min_frac_het_for_het`` is the main
        cross-patient knob.
    verbose
        Print run summary.

    Returns
    -------
    all_variant_germline_df, Sample_T_subset, germline_profile_meta
    """

    if experiment is None or str(experiment).strip() == "":
        raise ValueError("experiment must be provided.")
    if patient_id is None or str(patient_id).strip() == "":
        raise ValueError("patient_id must be provided.")

    patient_id_for_export = _safe_patient_export_id(experiment, patient_id)

    if verbose:
        print(f"Patient export ID: {patient_id_for_export}")

    all_variant_germline_df, Sample_T_subset, germline_profile_meta = (
        build_and_save_all_variant_germline_profile(
            Sample,
            DNA_Utils,
            patient_id=patient_id_for_export,
            annotation_df=annotation_df,
            known_het_variants=known_het_variants,
            keep_labels=keep_labels,
            out_dir=out_dir,
            descriptive_min_dp=descriptive_min_dp,
            placement_min_dp=placement_min_dp,
            min_frac_het_for_het=min_frac_het_for_het,
            max_frac_hom_for_het=max_frac_hom_for_het,
            max_frac_wt_for_het=max_frac_wt_for_het,
            min_called_cells_for_genotype=min_called_cells_for_genotype,
            verbose=verbose,
        )
    )

    germline = all_variant_germline_df.loc[
        all_variant_germline_df["usable_germline_het_seed"],
        "variant",
    ].tolist()

    clean_germline = all_variant_germline_df.loc[
        all_variant_germline_df["clean_germline_het_seed"],
        "variant",
    ].tolist()

    rescued_germline = all_variant_germline_df.loc[
        all_variant_germline_df["rescued_germline_het_seed"],
        "variant",
    ].tolist()

    ref = germline_profile_meta["ref"]

    germline_profile_meta = dict(germline_profile_meta)
    germline_profile_meta.update({
        "patient_id_for_export": patient_id_for_export,
        "germline": germline,
        "clean_germline": clean_germline,
        "rescued_germline": rescued_germline,
        "ref": ref,
    })

    if verbose:
        print()
        print("Reference profile:")
        print(ref)
        print()
        print(f"Clean germline HET seed:   {len(clean_germline)}")
        print(f"Rescued germline HET seed: {len(rescued_germline)}")
        print(f"Usable germline HET seed:  {len(germline)}")
        print(f"CSV: {germline_profile_meta['out_csv']}")

    return all_variant_germline_df, Sample_T_subset, germline_profile_meta


# -----------------------------------------------------------------------------
# Genotype-defined WT/clean-cell population + cross-population concordance
# -----------------------------------------------------------------------------


_EXCLUDE_CLONE_DF_COLUMNS = {
    "Variant\nname",
    "Variant name",
    "variant",
    "Variant",
    "variant_name",
    "Variant_name",
}

_EXCLUDE_CLONE_DF_PATTERNS = [
    r"^Missing GT clones",
    r"^Small subclones",
    r"^ADO clones",
    r"^FP clones",
    r"^missing$",
    r"^small$",
    r"^ADO$",
    r"^FP$",
]


def _get_clone_columns_from_clone_df(clone_df):
    """Return informative clone columns from a Mosaic VariantSubcloneTable
    dataframe, excluding variant-name and non-informative clone-summary columns.
    """
    if clone_df is None or not isinstance(clone_df, pd.DataFrame):
        return []

    clone_cols = []
    for c in clone_df.columns:
        c_str = str(c)
        if c_str in _EXCLUDE_CLONE_DF_COLUMNS:
            continue
        if any(re.search(pat, c_str, flags=re.IGNORECASE)
               for pat in _EXCLUDE_CLONE_DF_PATTERNS):
            continue
        clone_cols.append(c_str)
    return clone_cols


def _clone_barcode_table_from_labeled_sample(sample_like, clone_labels):
    """Build a barcode-level clone table from a Mosaic sample/assay carrying
    VariantSubcloneTable labels.

    Parameters
    ----------
    sample_like : Mosaic Sample-like object
        Object whose ``dna`` assay still has labels such as WT, TP53_Het, etc.
    clone_labels : iterable[str]
        Clone labels to recover barcode membership for.

    Returns
    -------
    clone_barcode_df : pd.DataFrame
        Columns: barcode, clone_label.
    cell_clone_labels : pd.Series
        Index = barcode, value = clone_label.
    """
    rows = []
    clone_labels = [str(x) for x in clone_labels]

    for lab in clone_labels:
        cells = []
        try:
            cells = sample_like.dna.barcodes({lab})
        except Exception:
            cells = []
        cells = np.asarray(cells).astype(str)
        for bc in cells:
            rows.append({"barcode": str(bc), "clone_label": lab})

    clone_barcode_df = pd.DataFrame(rows, columns=["barcode", "clone_label"])
    if clone_barcode_df.empty:
        cell_clone_labels = pd.Series(dtype=object, name="clone_label")
        cell_clone_labels.index = cell_clone_labels.index.astype(str)
        return clone_barcode_df, cell_clone_labels

    n_dup = int(clone_barcode_df["barcode"].duplicated().sum())
    if n_dup > 0:
        clone_barcode_df = clone_barcode_df.drop_duplicates("barcode", keep="first")

    cell_clone_labels = pd.Series(
        clone_barcode_df["clone_label"].astype(str).values,
        index=clone_barcode_df["barcode"].astype(str).values,
        name="clone_label",
    )
    return clone_barcode_df, cell_clone_labels

def build_wt_cell_subset(
    Sample_for_clustering,
    whitelist,
    *,
    DNA_Utils=None,
    ms_module=None,
    max_ado_score: float = 0.8,
    min_clone_cells: int = 5,
    total_barcodes: int = None,
    drop_labels=("missing", "small", "ADO", "FP"),
    verbose: bool = True,
):
    """Build a genotype-defined clean-cell ("WT") population, regardless of
    protein cell-type label, via the Mosaic VariantSubcloneTable workflow on the
    germline whitelist.

    Mirrors the standard subclone QC: cluster on the whitelist genotypes, then
    drop the cells the workflow flags as missing / small / ADO / FP. The
    survivors are the cells with clean, consistent germline genotypes — a much
    larger normal-evidence population than the protein-labelled lymphoid subset
    when cell-typing is sparse (e.g. the 9-cell MDS001 case).

    Clustering is run on a whitelist-restricted view of the sample (the workflow
    annotates every variant it is given, and the full panel exceeds the Varsome
    annotation cap); the resulting clean-cell selection is then applied to the
    full panel so all candidate variants remain available downstream. Depends on
    ``ms.workflows.VariantSubcloneTable`` and ``DNA_Utils.clones_to_dataframe``.

    Returns
    -------
    Sample_wt : the full-panel sample restricted to the clean clone cells
    clone_df  : the clone table from clones_to_dataframe. Barcode-level clone
                membership is stored in clone_df.attrs under
                'clone_barcode_df', 'cell_clone_labels', and
                'step2_clone_labels_used'.
    """
    import copy

    if ms_module is None:
        import missionbio.mosaic as ms_module

    if DNA_Utils is None or not hasattr(DNA_Utils, "clones_to_dataframe"):
        raise ValueError(
            "Pass DNA_Utils (exposing clones_to_dataframe) to build_wt_cell_subset."
        )

    # Cluster on the whitelist only. VariantSubcloneTable annotates every variant
    # in the sample it is given, and the full panel exceeds the Varsome cap, so
    # the clustering sample must be restricted to the whitelist (which is all
    # clones_to_dataframe uses anyway).
    present = [v for v in map(str, whitelist)
               if v in set(map(str, Sample_for_clustering.dna.ids()))]
    if len(present) == 0:
        raise ValueError("None of the whitelist variants are present in the sample.")

    clust = copy.copy(Sample_for_clustering)
    clust.dna = clust.dna[:, present]
    clust.dna.layers["NGT_FILTERED"] = clust.dna.layers["NGT"]

    wfv = ms_module.workflows.VariantSubcloneTable(clust)

    if total_barcodes is None:
        total_barcodes = len(Sample_for_clustering.dna.row_attrs["barcode"])
    min_clone_size = (min_clone_cells / total_barcodes) * 100.0

    clone_df = DNA_Utils.clones_to_dataframe(
        wfv,
        variants=present,
        max_ado_score=max_ado_score,
        min_clone_size=min_clone_size,
    )

    # Preserve barcode-level membership for the informative clones while the
    # VariantSubcloneTable labels are still attached to the clustering sample.
    # The returned clone_df is a variant x clone-signature table; by itself it
    # does not contain cell barcodes. Store the barcode table in attrs so the
    # higher-level step2 wrapper can expose it without changing this function's
    # public two-object return signature.
    step2_clone_labels = _get_clone_columns_from_clone_df(clone_df)
    clone_barcode_df, cell_clone_labels = _clone_barcode_table_from_labeled_sample(
        clust,
        step2_clone_labels,
    )
    clone_df.attrs["step2_clone_labels_used"] = step2_clone_labels
    clone_df.attrs["clone_barcode_df"] = clone_barcode_df
    clone_df.attrs["cell_clone_labels"] = cell_clone_labels

    # Collect the barcodes the workflow flagged as non-informative, then drop
    # those cells from a FULL-panel copy so every candidate variant remains
    # available for downstream structure scoring.
    junk = []
    for c in drop_labels:
        cells = clust.dna.barcodes({c})
        if len(cells) > 0:
            junk.extend(list(cells))

    Sample_wt = copy.copy(Sample_for_clustering)
    if len(junk) > 0:
        Sample_wt.dna = Sample_wt.dna.drop(np.array(junk))

    if verbose:
        print(f"  WT/clean-cell subset: {Sample_wt.dna.shape[0]} cells "
              f"(clustered on {len(present)} whitelist variants, "
              f"dropped {len(junk)} as {list(drop_labels)}); "
              f"all {Sample_wt.dna.shape[1]} panel variants retained.")

    return Sample_wt, clone_df


def population_structure_table(
    Sample_like,
    variants,
    *,
    label: str = "wt",
    min_dp: int = 5,
    min_het: float = 0.55,
    max_hom: float = 0.20,
    max_wt: float = 0.90,
    min_cells: int = 20,
):
    """Per-variant WT/HET/HOM call structure and donor genotype for an arbitrary
    cell population, with columns prefixed by ``label`` (e.g. ``wt_frac_HET``,
    ``wt_donor_genotype``). Uses the same structure logic and thresholds as the
    primary pass, so the two populations are directly comparable.
    """
    struct = compute_genotype_structure(Sample_like, variants, min_dp=min_dp)
    if struct.empty:
        return pd.DataFrame({"variant": []})

    struct["donor_genotype"] = struct.apply(
        lambda r: _donor_genotype_call(
            r.get("frac_WT"), r.get("frac_HET"), r.get("frac_HOM"),
            r.get("n_cells"),
            min_het=min_het, max_hom=max_hom, max_wt=max_wt, min_cells=min_cells,
        ),
        axis=1,
    )

    keep = ["variant", "n_cells", "frac_WT", "frac_HET", "frac_HOM",
            "het_af_median", "donor_genotype"]
    keep = [c for c in keep if c in struct.columns]
    return struct[keep].rename(
        columns={c: f"{label}_{c}" for c in keep if c != "variant"}
    )


def add_population_concordance(
    all_info,
    secondary_sample,
    *,
    label: str = "wt",
    variants=None,
    min_dp: int = 5,
    min_frac_het_for_het: float = 0.55,
    max_frac_hom_for_het: float = 0.20,
    max_frac_wt_for_het: float = 0.90,
    min_called_cells_for_genotype: int = 20,
    verbose: bool = True,
):
    """Attach a second cell population's call structure to the germline table,
    as its own columns, for per-patient decision-making — no single combined
    rule is imposed.

    Adds, all prefixed by ``label`` (e.g. ``wt_``):
      {label}_n_cells, {label}_frac_WT / _frac_HET / _frac_HOM,
      {label}_het_af_median, {label}_donor_genotype          (per variant)
    plus two informational booleans:
      {label}_germline_het_seed : the secondary population alone genotypes HET
      concordant_germline_het   : donor genotype is HET in BOTH populations

    The lymphoid seeds (clean / rescued / usable) are left untouched, and no
    ``confirmed`` column is created. You combine them per patient — typically
    trust ``concordant_germline_het`` when the lymphoid subset is well powered,
    and fall back to ``{label}_germline_het_seed`` when it is not. The printed
    summary reports both population sizes so that choice is informed.
    """
    all_info = all_info.copy()
    if variants is None:
        variants = all_info["variant"].astype(str).tolist()

    sec = population_structure_table(
        secondary_sample, variants, label=label, min_dp=min_dp,
        min_het=min_frac_het_for_het, max_hom=max_frac_hom_for_het,
        max_wt=max_frac_wt_for_het, min_cells=min_called_cells_for_genotype,
    )
    all_info = all_info.merge(sec, on="variant", how="left")

    sec_dg = f"{label}_donor_genotype"
    wt_het = (all_info[sec_dg].eq("het")
              if sec_dg in all_info.columns
              else pd.Series(False, index=all_info.index)).fillna(False)
    all_info[f"{label}_germline_het_seed"] = wt_het

    primary = all_info.get("donor_genotype")
    primary_het = (primary.eq("het").fillna(False)
                   if primary is not None
                   else pd.Series(False, index=all_info.index))
    all_info["concordant_germline_het"] = primary_het & wt_het

    if verbose:
        try:
            wt_cells = int(secondary_sample.dna.shape[0])
        except Exception:
            wt_cells = "?"
        lymph_cells = (int(all_info["n_T_cells_in_subset"].iloc[0])
                       if "n_T_cells_in_subset" in all_info.columns and len(all_info)
                       else "?")
        usable = all_info.get("usable_germline_het_seed")
        usable_n = int(usable.fillna(False).sum()) if usable is not None else 0

        print(f"  Population sizes: lymphoid={lymph_cells} cells, "
              f"{label}={wt_cells} cells")
        print(f"    lymphoid usable germline seeds : {usable_n}")
        print(f"    {label} HET seeds               : {int(wt_het.sum())}")
        print(f"    concordant (HET in both)        : "
              f"{int(all_info['concordant_germline_het'].sum())}")
        print("  Per-patient recipes (you choose):")
        print("    well-powered lymphoid -> usable_germline_het_seed & "
              "concordant_germline_het")
        print(f"    under-powered lymphoid -> {label}_germline_het_seed")
        print(f"    union               -> usable_germline_het_seed | "
              f"{label}_germline_het_seed")

    return all_info


# -----------------------------------------------------------------------------
# Cohort-level germline-HET threshold calibration
# -----------------------------------------------------------------------------
#
# Pools the per-patient *_variants_germline_data.csv files, anchors on the
# known-germline whitelist instances to learn what a genuine germline HET looks
# like across the cohort, corroborates with cross-donor recurrence, and proposes
# sample-independent thresholds (frac_HET / frac_HOM / frac_WT and a HET-AF
# window). No hard ground truth exists, so "accurate" is empirical: whitelist
# sites called HET are the positive anchor; recurrence corroborates, with the
# standing caveat that reproducible artefacts (chrX non-PAR paralogues, indel AF
# miscalibration) also recur and must be read alongside frac_HOM and AF shape.

CSV_GLOB = "*_variants_germline_data.csv"

# logical feature -> candidate column names, first match wins
# Prefer the placement_* columns when present because donor_genotype is derived
# from the DP>=placement_min_dp genotype structure. Fall back to the broader
# descriptive columns for older CSV exports.
COL_CANDIDATES = {
    "variant": ["variant", "variant_id", "id"],
    "frac_het": ["placement_frac_HET", "frac_HET", "frac_het"],
    "frac_hom": ["placement_frac_HOM", "frac_HOM_ALT", "frac_HOM", "frac_hom_alt", "frac_hom"],
    "frac_wt": ["placement_frac_WT", "frac_WT", "frac_wt"],
    "het_af_median": [
        "placement_HET_AF_median",
        "AF_HET_median",
        "het_af_median",
        "median_af",
        "pooled_af",
        "pooled_het_af",
        "af_median",
    ],
    "n_cells": ["placement_n_cells", "n_valid_called", "n_cells", "n_called_cells", "n_cells_called"],
    "donor_genotype": ["donor_genotype"],
    "usable_seed": ["usable_germline_het_seed"],
    "clean_seed": ["clean_germline_het_seed"],
    "rescued_seed": ["rescued_germline_het_seed"],
    "concordant_seed": ["concordant_germline_het"],
    "known_het": [
        "is_known_het_whitelist",
        "is_whitelist_for_reference",
        "known_het",
        "is_known_het",
        "in_whitelist",
        "whitelist",
    ],
    "status": ["status"],
    "review_verdict": ["review_verdict"],
    "broad_class": ["germline_broad_class"],
    "pathogenic": ["is_pathogenic_excluded"],
    "blacklisted": ["is_blacklisted"],
    "af_z": ["z_robust_sd", "af_z", "z", "z_af", "af_zscore"],
}

# genotype labels that mean the variant was not adequately observed
NOT_PRESENT = {
    "uncalled",
    "low_coverage",
    "low_occupancy_review",
    "het_low_occupancy_review",
    "low_coverage_review",
}

# classes that should never be gained by a purely numerical calibration rule
BAD_GERMLINE_CLASSES = {
    "excluded_blacklisted",
    "excluded_pathogenic",
    "indel_af_unreliable",
    "likely_hom_alt",
    "likely_hom_ref",
    "likely_hom_or_ref",
    "overdispersed",
}

CURRENT_MIN_FRAC_HET = 0.55  # value presently in the pipeline, for reference
CURRENT_MAX_FRAC_HOM = 0.20
CURRENT_MAX_FRAC_WT = 0.90


def resolve_calibration_columns(df):
    """Map logical feature names to the actual columns present in df."""
    colmap = {}
    for logical, candidates in COL_CANDIDATES.items():
        for c in candidates:
            if c in df.columns:
                colmap[logical] = c
                break
    return colmap


def load_germline_cohort(csv_dir=GERMLINE_PATTERN_DIR, pattern=CSV_GLOB):
    """Pool all per-patient germline CSVs, tagging each row with its sample."""
    paths = sorted(glob.glob(os.path.join(csv_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSVs matching {pattern} in {csv_dir}")
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["sample_id"] = os.path.basename(p).replace(
            "_variants_germline_data.csv", "")
        frames.append(d)
        print(f"  loaded {os.path.basename(p)}: {d.shape[0]} variants")
    pooled = pd.concat(frames, ignore_index=True)
    print(f"\nPooled: {pooled.shape[0]} variant-instances "
          f"across {len(paths)} samples")
    return pooled, paths


def _cal_col(df, colmap, logical):
    """Series for a logical feature, or None if the column is absent."""
    col = colmap.get(logical)
    return df[col] if col is not None and col in df.columns else None


def _numeric_cal_col(df, colmap, logical, default=np.nan):
    """Numeric Series for a logical feature, robust to missing columns."""
    s = _cal_col(df, colmap, logical)
    if s is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(s, errors="coerce")


def _bool_cal_col(df, colmap, logical, default=False):
    """Boolean Series for a logical feature, robust to missing columns."""
    s = _cal_col(df, colmap, logical)
    if s is None:
        return pd.Series(default, index=df.index, dtype=bool)
    return _coerce_bool_series(s).reindex(df.index).fillna(default).astype(bool)


def _string_cal_col(df, colmap, logical, default=""):
    """String Series for a logical feature, robust to missing columns."""
    s = _cal_col(df, colmap, logical)
    if s is None:
        return pd.Series(default, index=df.index, dtype=object)
    return s.fillna(default).astype(str)


def _present_mask(df, colmap):
    dg = _cal_col(df, colmap, "donor_genotype")
    if dg is not None:
        return ~dg.isin(NOT_PRESENT) & dg.notna()
    nc = _cal_col(df, colmap, "n_cells")
    if nc is not None:
        return pd.to_numeric(nc, errors="coerce").fillna(0) >= 1
    return pd.Series(True, index=df.index)


def _het_mask(df, colmap):
    dg = _cal_col(df, colmap, "donor_genotype")
    if dg is not None:
        return dg.eq("het")
    seed = _cal_col(df, colmap, "usable_seed")
    if seed is not None:
        return _coerce_bool_series(seed).fillna(False).astype(bool)
    raise ValueError("Need donor_genotype or usable_germline_het_seed to call HET.")


def _not_excluded_mask(df, colmap):
    pathogenic = _bool_cal_col(df, colmap, "pathogenic")
    blacklisted = _bool_cal_col(df, colmap, "blacklisted")
    broad_class = _string_cal_col(df, colmap, "broad_class")
    return (~pathogenic) & (~blacklisted) & (~broad_class.isin(BAD_GERMLINE_CLASSES))


def germline_recurrence(pooled, colmap):
    """Per-variant cross-donor recurrence summary."""
    var = _cal_col(pooled, colmap, "variant")
    if var is None:
        raise ValueError("No variant identifier column found.")
    work = pd.DataFrame({
        "variant": var,
        "sample_id": pooled["sample_id"],
        "present": _present_mask(pooled, colmap).values,
        "het": _het_mask(pooled, colmap).values,
    })
    rec = work.groupby("variant").agg(
        n_samples=("sample_id", "nunique"),
        n_present=("present", "sum"),
        n_het=("het", "sum"),
    ).reset_index()
    rec["het_call_rate"] = rec["n_het"] / rec["n_present"].replace(0, np.nan)
    return rec.sort_values(["n_het", "het_call_rate"], ascending=False)


def germline_anchor_frame(pooled, colmap):
    """Whitelist instances actually called HET (positive anchor).

    Falls back to recurrent HET instances if no/sparse whitelist column exists.
    Excluded/pathogenic/blacklisted/bad-class rows are not allowed to define the
    anchor distribution.
    """
    het = _het_mask(pooled, colmap)
    allowed = _not_excluded_mask(pooled, colmap)
    known = _cal_col(pooled, colmap, "known_het")
    if known is not None:
        known = _coerce_bool_series(known).fillna(False).astype(bool)
        anchor = pooled[known & het & allowed]
        if len(anchor) >= 20:
            return anchor, "whitelist HET instances"
        print(f"  [warn] only {len(anchor)} whitelist HET instances; "
              f"augmenting with recurrent HETs")
    rec = germline_recurrence(pooled[allowed].copy(), colmap)
    recurrent = set(rec.loc[rec["n_het"] >= 3, "variant"])
    var = _cal_col(pooled, colmap, "variant")
    anchor = pooled[allowed & het & var.isin(recurrent)]
    return anchor, "recurrent HET instances (no/sparse whitelist)"


def germline_percentile_table(anchor, colmap):
    """Empirical percentiles of each feature in the anchor set."""
    feats = {
        "frac_het": [1, 2.5, 5, 10, 25, 50],
        "frac_hom": [50, 75, 90, 95, 97.5, 99],
        "frac_wt":  [50, 75, 90, 95, 97.5, 99],
        "het_af_median": [1, 2.5, 5, 50, 95, 97.5, 99],
        "af_z": [50, 90, 95, 97.5, 99],
    }
    rows = []
    for logical, pcts in feats.items():
        s = _numeric_cal_col(anchor, colmap, logical).dropna()
        if s.empty:
            continue
        for p in pcts:
            rows.append({"feature": logical, "percentile": p,
                         "value": np.percentile(s, p)})
    return pd.DataFrame(rows)


def propose_germline_thresholds(anchor, colmap):
    """Percentile-derived thresholds for the clean canonical HET tier.

    This function intentionally learns only the clean/canonical tier from the
    whitelist/recurrent HET anchor distribution. The practical usable tier is
    learned separately by ``calibrate_practical_germline_thresholds`` using a
    broader positive anchor and explicit bad-class penalties.
    """
    def pc(logical, p):
        s = _numeric_cal_col(anchor, colmap, logical).dropna()
        return float(np.percentile(s, p)) if len(s) else None

    clean_min_frac_het = pc("frac_het", 5)
    clean_max_frac_hom = pc("frac_hom", 95)
    clean_max_frac_wt = pc("frac_wt", 95)
    clean_af_lower = pc("het_af_median", 2.5)
    clean_af_upper = pc("het_af_median", 97.5)
    max_af_z = pc("af_z", 95)

    return {
        # Backwards-compatible names = clean canonical tier.
        "min_frac_het": clean_min_frac_het,
        "max_frac_hom": clean_max_frac_hom,
        "max_frac_wt": clean_max_frac_wt,
        "af_lower": clean_af_lower,
        "af_upper": clean_af_upper,
        "max_af_z": max_af_z,

        # Explicit clean tier.
        "clean_min_frac_het": clean_min_frac_het,
        "clean_max_frac_hom": clean_max_frac_hom,
        "clean_max_frac_wt": clean_max_frac_wt,
        "clean_af_lower": clean_af_lower,
        "clean_af_upper": clean_af_upper,
        "clean_max_af_z": max_af_z,
    }


def _bad_practical_mask(df, colmap):
    """Rows that should never be gained by practical threshold calibration."""
    broad = _string_cal_col(df, colmap, "broad_class")
    status = _string_cal_col(df, colmap, "status")
    review = _string_cal_col(df, colmap, "review_verdict")
    dg = _string_cal_col(df, colmap, "donor_genotype")

    pathogenic = _bool_cal_col(df, colmap, "pathogenic")
    blacklisted = _bool_cal_col(df, colmap, "blacklisted")

    return (
        pathogenic
        | blacklisted
        | broad.isin(BAD_GERMLINE_CLASSES)
        | status.eq("overdispersed")
        | review.eq("review_hom_or_ref")
        | dg.isin(["hom_alt", "hom_ref"])
    )


def germline_practical_anchor_frame(pooled, colmap):
    """Broader empirical positive anchor for practical usable HET thresholds.

    The practical anchor is intentionally broader than the clean anchor. It
    includes currently usable germline HETs, clean HETs, rescued/skewed HETs,
    and whitelist HETs, provided they are not excluded or bad-class rows.
    """
    het = _het_mask(pooled, colmap)
    bad = _bad_practical_mask(pooled, colmap)

    known = _bool_cal_col(pooled, colmap, "known_het")
    usable = _bool_cal_col(pooled, colmap, "usable_seed")
    clean = _bool_cal_col(pooled, colmap, "clean_seed")
    rescued = _bool_cal_col(pooled, colmap, "rescued_seed")

    broad = _string_cal_col(pooled, colmap, "broad_class")
    status = _string_cal_col(pooled, colmap, "status")
    review = _string_cal_col(pooled, colmap, "review_verdict")

    rescued_like = (
        rescued
        | broad.isin([
            "rescued_germline_het_candidate",
            "rescued_germline_het_whitelist_low_evidence",
        ])
        | (status.eq("af_skewed") & review.eq("likely_usable_het"))
    )

    anchor_mask = het & ~bad & (usable | clean | rescued_like | known)
    anchor = pooled.loc[anchor_mask].copy()

    return anchor, "current usable + whitelist + rescued/skewed HET instances"


def calibrate_practical_germline_thresholds(
    pooled,
    practical_anchor,
    colmap,
    *,
    recall_floor: float = 0.995,
    current_reject_floor: int = 0,
    min_het_percentiles=None,
    max_hom_percentiles=None,
    max_wt_percentiles=None,
):
    """Empirically tune practical genotype-structure thresholds.

    This v1 calibrator removes hand-seeded numeric cutoffs. Candidate
    thresholds are generated only from percentiles of the practical positive
    anchor distribution:

      - ``usable_min_frac_het`` = a lower percentile of anchor frac_HET.
      - ``usable_max_frac_hom`` = an upper percentile of anchor frac_HOM.
      - ``usable_max_frac_wt``  = an upper percentile of anchor frac_WT.

    The selected operating point is the most stringent empirical percentile
    combination that satisfies the requested anchor recall/current-retention
    constraint, while minimising bad-class and non-anchor pass-through. AF is
    deliberately not used here; AF defines the clean canonical tier only.
    Rescued skewed HETs are handled by review/genotype support.
    """
    fh = _numeric_cal_col(pooled, colmap, "frac_het")
    fhom = _numeric_cal_col(pooled, colmap, "frac_hom")
    fwt = _numeric_cal_col(pooled, colmap, "frac_wt")

    valid = fh.notna() & fhom.notna() & fwt.notna()
    bad = _bad_practical_mask(pooled, colmap)
    allowed = _not_excluded_mask(pooled, colmap)
    donor_het = _het_mask(pooled, colmap)
    current = _bool_cal_col(pooled, colmap, "usable_seed")

    anchor_mask = pd.Series(False, index=pooled.index)
    anchor_mask.loc[practical_anchor.index] = True
    anchor_mask &= valid

    a_fh = fh.loc[anchor_mask].dropna()
    a_fhom = fhom.loc[anchor_mask].dropna()
    a_fwt = fwt.loc[anchor_mask].dropna()

    if len(a_fh) == 0 or len(a_fhom) == 0 or len(a_fwt) == 0:
        raise ValueError(
            "Practical anchor has insufficient frac_HET / frac_HOM / frac_WT values."
        )

    # Percentile choices, not numeric cutoffs. These are deliberately exposed
    # in the output table so the operating point is auditable.
    if min_het_percentiles is None:
        min_het_percentiles = [0, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    if max_hom_percentiles is None:
        max_hom_percentiles = [90, 95, 97.5, 99, 99.5, 99.75, 99.9, 100]
    if max_wt_percentiles is None:
        max_wt_percentiles = [90, 95, 97.5, 99, 99.5, 99.75, 99.9, 100]

    def _pc(values, p):
        return float(np.percentile(values, p))

    rows = []
    n_anchor = int(anchor_mask.sum())

    for p_min_het in min_het_percentiles:
        min_het = _pc(a_fh, p_min_het)
        for p_max_hom in max_hom_percentiles:
            max_hom = _pc(a_fhom, p_max_hom)
            for p_max_wt in max_wt_percentiles:
                max_wt = _pc(a_fwt, p_max_wt)

                rule = (
                    valid
                    & donor_het
                    & (fh >= min_het)
                    & (fhom <= max_hom)
                    & (fwt <= max_wt)
                )

                anchor_retained = int((rule & anchor_mask).sum())
                anchor_recall = anchor_retained / n_anchor if n_anchor else np.nan

                bad_passing_raw = int((rule & bad).sum())
                hard_excluded_passing_raw = int((rule & ~allowed).sum())
                nonanchor_allowed_passing = int((rule & allowed & ~anchor_mask).sum())
                current_rejected = int((current & ~rule).sum())
                current_retained = int((current & rule).sum())
                total_passing_raw = int(rule.sum())
                total_passing_after_hard_exclusions = int((rule & allowed).sum())

                labelled_precision = (
                    anchor_retained / (anchor_retained + bad_passing_raw)
                    if (anchor_retained + bad_passing_raw) > 0
                    else np.nan
                )

                rows.append({
                    "min_het_percentile": float(p_min_het),
                    "max_hom_percentile": float(p_max_hom),
                    "max_wt_percentile": float(p_max_wt),
                    "usable_min_frac_het": float(min_het),
                    "usable_max_frac_hom": float(max_hom),
                    "usable_max_frac_wt": float(max_wt),
                    "practical_anchor_n": n_anchor,
                    "practical_anchor_retained": anchor_retained,
                    "practical_anchor_recall": float(anchor_recall),
                    "labelled_precision_vs_bad": float(labelled_precision),
                    "bad_passing_raw": bad_passing_raw,
                    "hard_excluded_passing_raw": hard_excluded_passing_raw,
                    "nonanchor_allowed_passing": nonanchor_allowed_passing,
                    "current_rejected": current_rejected,
                    "current_retained": current_retained,
                    "total_passing_raw": total_passing_raw,
                    "total_passing_after_hard_exclusions": total_passing_after_hard_exclusions,
                })

    search = pd.DataFrame(rows)

    eligible = search[
        (search["practical_anchor_recall"] >= recall_floor)
        & (search["current_rejected"] <= current_reject_floor)
    ].copy()

    if eligible.empty:
        eligible = search.copy()

    # No hidden weighted score: the sort order is the operating-point policy.
    # First satisfy positive-anchor recall/current retention, then minimise bad
    # pass-through and new non-anchor pass-through. Ties choose the most compact
    # final set, then the highest observed recall.
    best_row = (
        eligible.sort_values(
            [
                "bad_passing_raw",
                "nonanchor_allowed_passing",
                "total_passing_after_hard_exclusions",
                "practical_anchor_recall",
                "min_het_percentile",
                "max_hom_percentile",
                "max_wt_percentile",
            ],
            ascending=[True, True, True, False, False, True, True],
        )
        .iloc[0]
    )

    practical_thresholds = {
        "usable_min_frac_het": float(best_row["usable_min_frac_het"]),
        "usable_max_frac_hom": float(best_row["usable_max_frac_hom"]),
        "usable_max_frac_wt": float(best_row["usable_max_frac_wt"]),
        "usable_min_frac_het_percentile": float(best_row["min_het_percentile"]),
        "usable_max_frac_hom_percentile": float(best_row["max_hom_percentile"]),
        "usable_max_frac_wt_percentile": float(best_row["max_wt_percentile"]),
        "usable_anchor_recall": float(best_row["practical_anchor_recall"]),
        "usable_labelled_precision_vs_bad": float(best_row["labelled_precision_vs_bad"]),
        "usable_bad_passing_raw": int(best_row["bad_passing_raw"]),
        "usable_hard_excluded_passing_raw": int(best_row["hard_excluded_passing_raw"]),
        "usable_nonanchor_allowed_passing": int(best_row["nonanchor_allowed_passing"]),
        "usable_current_rejected": int(best_row["current_rejected"]),
        "usable_current_retained": int(best_row["current_retained"]),
        "usable_total_passing_raw": int(best_row["total_passing_raw"]),
        "usable_total_passing_after_hard_exclusions": int(best_row["total_passing_after_hard_exclusions"]),
        "usable_selection_policy": (
            "percentile_grid; require anchor_recall>=recall_floor and "
            "current_rejected<=current_reject_floor; minimise bad_passing_raw, "
            "then nonanchor_allowed_passing, then total passing"
        ),
    }

    search = search.sort_values(
        [
            "bad_passing_raw",
            "nonanchor_allowed_passing",
            "total_passing_after_hard_exclusions",
            "practical_anchor_recall",
        ],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    return practical_thresholds, search


def evaluate_min_frac_het(pooled, anchor, colmap, grid=None):
    """Anchor recall vs permissiveness on the rest, across min_frac_het."""
    if grid is None:
        grid = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70]
    af = _numeric_cal_col(anchor, colmap, "frac_het").dropna()
    all_fh = _numeric_cal_col(pooled, colmap, "frac_het")
    het = _het_mask(pooled, colmap)
    known = _bool_cal_col(pooled, colmap, "known_het")
    allowed = _not_excluded_mask(pooled, colmap)
    rows = []
    for thr in grid:
        recall = float((af >= thr).mean()) if len(af) else np.nan
        nonwl = allowed & het & ~known & (all_fh >= thr)
        rows.append({"min_frac_het": thr,
                     "anchor_recall": round(recall, 3),
                     "nonwhitelist_het_passing": int(nonwl.sum())})
    return pd.DataFrame(rows)


def apply_calibrated_germline_rules(pooled, thresholds, colmap=None):
    """Add clean/practical calibrated germline flags to a pooled dataframe.

    The practical flag intentionally keeps rescued AF-skewed HETs and existing
    usable/concordant HETs, while preventing numerical thresholds from gaining
    blacklisted, pathogenic, HOM-like, indel-unreliable, or overdispersed rows.
    """
    df = pooled.copy()
    if colmap is None:
        colmap = resolve_calibration_columns(df)

    frac_het = _numeric_cal_col(df, colmap, "frac_het")
    frac_hom = _numeric_cal_col(df, colmap, "frac_hom")
    frac_wt = _numeric_cal_col(df, colmap, "frac_wt")
    af_med = _numeric_cal_col(df, colmap, "het_af_median")

    donor_het = _het_mask(df, colmap)
    usable_seed = _bool_cal_col(df, colmap, "usable_seed")
    clean_seed = _bool_cal_col(df, colmap, "clean_seed")
    rescued_seed = _bool_cal_col(df, colmap, "rescued_seed")
    concordant_seed = _bool_cal_col(df, colmap, "concordant_seed")
    status = _string_cal_col(df, colmap, "status")
    review = _string_cal_col(df, colmap, "review_verdict")
    allowed = _not_excluded_mask(df, colmap)

    clean_min_frac_het = thresholds.get("clean_min_frac_het", thresholds.get("min_frac_het"))
    clean_max_frac_hom = thresholds.get("clean_max_frac_hom", thresholds.get("max_frac_hom"))
    clean_max_frac_wt = thresholds.get("clean_max_frac_wt", thresholds.get("max_frac_wt"))
    clean_af_lower = thresholds.get("clean_af_lower", thresholds.get("af_lower"))
    clean_af_upper = thresholds.get("clean_af_upper", thresholds.get("af_upper"))

    usable_min_frac_het = thresholds.get("usable_min_frac_het", clean_min_frac_het)
    usable_max_frac_hom = thresholds.get("usable_max_frac_hom", clean_max_frac_hom)
    usable_max_frac_wt = thresholds.get("usable_max_frac_wt", clean_max_frac_wt)

    clean = allowed & donor_het
    if clean_min_frac_het is not None:
        clean &= frac_het >= clean_min_frac_het
    if clean_max_frac_hom is not None:
        clean &= frac_hom <= clean_max_frac_hom
    if clean_max_frac_wt is not None:
        clean &= frac_wt <= clean_max_frac_wt
    if clean_af_lower is not None and clean_af_upper is not None:
        clean &= af_med.between(clean_af_lower, clean_af_upper).fillna(False)

    structure_supported = allowed & donor_het
    if usable_min_frac_het is not None:
        structure_supported &= frac_het >= usable_min_frac_het
    if usable_max_frac_hom is not None:
        structure_supported &= frac_hom <= usable_max_frac_hom
    if usable_max_frac_wt is not None:
        structure_supported &= frac_wt <= usable_max_frac_wt

    rescued_skewed = (
        allowed
        & donor_het
        & (
            rescued_seed
            | (status.eq("af_skewed") & review.eq("likely_usable_het"))
            | _string_cal_col(df, colmap, "broad_class").isin([
                "rescued_germline_het_candidate",
                "rescued_germline_het_whitelist_low_evidence",
            ])
        )
    )

    practical = allowed & donor_het & (
        clean
        | rescued_skewed
        | (usable_seed & (structure_supported | clean_seed | rescued_seed))
        | concordant_seed
    )

    df["calibrated_clean_het_seed"] = clean.fillna(False).astype(bool)
    df["calibrated_rescued_het_seed"] = rescued_skewed.fillna(False).astype(bool)
    df["calibrated_usable_het_seed"] = practical.fillna(False).astype(bool)
    df["calibrated_rejected_het_seed"] = (
        usable_seed & ~df["calibrated_usable_het_seed"]
    ).fillna(False).astype(bool)

    fail_parts = []
    fail_parts.append(np.where(~allowed, "excluded_or_bad_class", ""))
    fail_parts.append(np.where(~donor_het, "not_donor_het", ""))
    if usable_min_frac_het is not None:
        fail_parts.append(np.where(frac_het < usable_min_frac_het, "low_frac_HET", ""))
    if usable_max_frac_hom is not None:
        fail_parts.append(np.where(frac_hom > usable_max_frac_hom, "high_frac_HOM", ""))
    if usable_max_frac_wt is not None:
        fail_parts.append(np.where(frac_wt > usable_max_frac_wt, "high_frac_WT", ""))
    fail_arr = np.vstack(fail_parts).T
    df["calibrated_reject_reasons"] = [
        ";".join([x for x in row if x]) for row in fail_arr
    ]

    return df


def germline_counts_per_sample(pooled, thresholds, colmap=None):
    """Summarise current vs calibrated germline calls per sample.

    ``n_calibrated_germline_het`` is the practical two-tier usable call, not the
    stricter clean-only canonical HET count.
    """
    if colmap is None:
        colmap = resolve_calibration_columns(pooled)
    d = apply_calibrated_germline_rules(pooled, thresholds, colmap=colmap)
    current = _bool_cal_col(d, colmap, "usable_seed")
    d["_current_germline_het"] = current

    out = (
        d.groupby("sample_id")
        .agg(
            n_variants=("variant", "size"),
            n_current_germline_het=("_current_germline_het", "sum"),
            n_calibrated_clean_germline_het=("calibrated_clean_het_seed", "sum"),
            n_calibrated_rescued_germline_het=("calibrated_rescued_het_seed", "sum"),
            n_calibrated_germline_het=("calibrated_usable_het_seed", "sum"),
            n_rejected_current_germline_het=("calibrated_rejected_het_seed", "sum"),
        )
        .reset_index()
    )
    out["delta"] = (
        out["n_calibrated_germline_het"]
        - out["n_current_germline_het"]
    )
    return out.sort_values("sample_id").reset_index(drop=True)


def _plot_germline_calibration(anchor, pooled, colmap, thresholds, out_png):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    fh = _numeric_cal_col(anchor, colmap, "frac_het").dropna()
    if len(fh):
        ax.hist(fh, bins=30, color="#4C72B0", alpha=0.85)
    if thresholds.get("clean_min_frac_het") is not None:
        ax.axvline(thresholds["clean_min_frac_het"], color="k", ls="--",
                   label=f"clean {thresholds['clean_min_frac_het']:.2f}")
    if thresholds.get("usable_min_frac_het") is not None:
        ax.axvline(thresholds["usable_min_frac_het"], color="0.35", ls="-.",
                   label=f"usable {thresholds['usable_min_frac_het']:.2f}")
    ax.axvline(CURRENT_MIN_FRAC_HET, color="crimson", ls=":",
               label=f"current {CURRENT_MIN_FRAC_HET:.2f}")
    excl = float((fh < CURRENT_MIN_FRAC_HET).mean()) * 100 if len(fh) else 0
    ax.set_title(f"Anchor frac_HET  (current floor drops {excl:.0f}% of anchors)")
    ax.set_xlabel("frac_HET")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    fhom = _numeric_cal_col(anchor, colmap, "frac_hom").dropna()
    if len(fhom):
        ax.hist(fhom, bins=30, color="#55A868", alpha=0.85)
    if thresholds.get("clean_max_frac_hom") is not None:
        ax.axvline(thresholds["clean_max_frac_hom"], color="k", ls="--", label="clean")
    if thresholds.get("usable_max_frac_hom") is not None:
        ax.axvline(thresholds["usable_max_frac_hom"], color="0.35", ls="-.", label="usable")
    ax.set_title("Anchor frac_HOM/HOM_ALT (should sit near 0)")
    ax.set_xlabel("frac_HOM / frac_HOM_ALT")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    afm = _numeric_cal_col(anchor, colmap, "het_af_median").dropna()
    if len(afm):
        ax.hist(afm, bins=30, color="#C44E52", alpha=0.85)
        for k in ("clean_af_lower", "clean_af_upper"):
            if thresholds.get(k) is not None:
                ax.axvline(thresholds[k], color="k", ls="--")
        ax.set_title("Anchor HET-AF median (clean tier only)")
        ax.set_xlabel("AF (%)")
    else:
        ax.text(0.5, 0.5, "het_af_median not resolved", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="0.4")
        ax.set_title("Anchor HET-AF median (unavailable)")

    ax = axes[1, 1]
    fh_all = _numeric_cal_col(pooled, colmap, "frac_het")
    nc_all = _numeric_cal_col(pooled, colmap, "n_cells")
    het = _het_mask(pooled, colmap)
    if nc_all.notna().any():
        ax.scatter(nc_all[het], fh_all[het], s=8, alpha=0.4, color="#8172B3")
        ax.set_xscale("log")
        ax.set_title("frac_HET vs cells (ADO spread = case for usable tier)")
        ax.set_xlabel("n_cells (log)")
        ax.set_ylabel("frac_HET")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"\nFigure: {out_png}")


def calibrate_germline_thresholds(csv_dir=GERMLINE_PATTERN_DIR, out_dir="."):
    """Pool per-patient germline CSVs and calibrate clean + practical HET thresholds.

    Returns ``(pooled, recurrence, anchor, thresholds, curve)`` and writes:
      - ``cohort_pooled.csv``
      - ``cohort_recurrence.csv``
      - ``cohort_per_sample_germline_counts.csv``
      - ``cohort_practical_threshold_grid_search.csv``
      - ``cohort_threshold_calibration.png``

    The returned ``pooled`` dataframe contains calibrated flags:
      - ``calibrated_clean_het_seed``: canonical near-50% clean HETs.
      - ``calibrated_rescued_het_seed``: rescued/skewed usable HETs.
      - ``calibrated_usable_het_seed``: practical downstream set.
    """
    pooled, _ = load_germline_cohort(csv_dir)
    colmap = resolve_calibration_columns(pooled)

    print("\nResolved columns:")
    for k, v in colmap.items():
        print(f"  {k:14s} -> {v}")

    missing = [k for k in ("variant", "frac_het", "frac_hom", "frac_wt")
               if k not in colmap]
    if missing:
        print(f"\n[warn] missing expected columns: {missing} "
              f"(edit COL_CANDIDATES)")

    rec = germline_recurrence(pooled, colmap)

    # Clean anchor: whitelist/recurrent canonical HET distribution.
    anchor, source = germline_anchor_frame(pooled, colmap)
    print(f"\nClean anchor: {len(anchor)} instances ({source})")

    pct = germline_percentile_table(anchor, colmap)
    print("\nClean-anchor percentile table:")
    print(pct.to_string(index=False))

    clean_thr = propose_germline_thresholds(anchor, colmap)

    # Practical anchor: current usable + whitelist + rescued/skewed HETs.
    practical_anchor, practical_source = germline_practical_anchor_frame(
        pooled,
        colmap,
    )
    print(f"\nPractical anchor: {len(practical_anchor)} instances "
          f"({practical_source})")

    practical_pct = germline_percentile_table(practical_anchor, colmap)
    print("\nPractical-anchor percentile table:")
    print(practical_pct.to_string(index=False))

    bad_for_pct = pooled.loc[_bad_practical_mask(pooled, colmap)].copy()
    bad_pct = germline_percentile_table(bad_for_pct, colmap)
    if not bad_pct.empty:
        print("\nBad-class percentile table:")
        print(bad_pct.to_string(index=False))

    practical_thr, practical_search = calibrate_practical_germline_thresholds(
        pooled,
        practical_anchor,
        colmap,
        recall_floor=0.995,
        current_reject_floor=0,
    )

    thr = dict(clean_thr)
    thr.update(practical_thr)

    print("\nClean canonical HET thresholds:")
    for k in ["clean_min_frac_het", "clean_max_frac_hom", "clean_max_frac_wt",
              "clean_af_lower", "clean_af_upper", "clean_max_af_z"]:
        v = thr.get(k)
        print(f"  {k:28s} = {v:.3f}" if v is not None else f"  {k:28s} = n/a")

    print("\nEmpirically selected practical usable HET thresholds:")
    print("  selected from practical-anchor percentiles, not hand-set numeric cutoffs")
    for k in [
        "usable_min_frac_het",
        "usable_min_frac_het_percentile",
        "usable_max_frac_hom",
        "usable_max_frac_hom_percentile",
        "usable_max_frac_wt",
        "usable_max_frac_wt_percentile",
        "usable_anchor_recall",
        "usable_labelled_precision_vs_bad",
        "usable_bad_passing_raw",
        "usable_hard_excluded_passing_raw",
        "usable_nonanchor_allowed_passing",
        "usable_current_rejected",
        "usable_current_retained",
        "usable_total_passing_after_hard_exclusions",
    ]:
        v = thr.get(k)
        if isinstance(v, float):
            print(f"  {k:38s} = {v:.3f}")
        else:
            print(f"  {k:38s} = {v}")
    print("  AF window is clean-tier only; rescued skewed HETs are retained "
          "by review/genotype support.")

    curve = evaluate_min_frac_het(pooled, anchor, colmap)
    print("\nmin_frac_het calibration (clean-anchor recall vs permissiveness):")
    print(curve.to_string(index=False))

    pooled = apply_calibrated_germline_rules(pooled, thr, colmap=colmap)
    per_sample = germline_counts_per_sample(pooled, thr, colmap=colmap)

    print("\nPer-sample current vs calibrated practical germline counts:")
    print(per_sample.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    pooled.to_csv(os.path.join(out_dir, "cohort_pooled.csv"), index=False)
    rec.to_csv(os.path.join(out_dir, "cohort_recurrence.csv"), index=False)
    per_sample.to_csv(
        os.path.join(out_dir, "cohort_per_sample_germline_counts.csv"),
        index=False,
    )
    practical_search.to_csv(
        os.path.join(out_dir, "cohort_practical_threshold_grid_search.csv"),
        index=False,
    )
    practical_pct.to_csv(
        os.path.join(out_dir, "cohort_practical_anchor_percentiles.csv"),
        index=False,
    )
    if not bad_pct.empty:
        bad_pct.to_csv(
            os.path.join(out_dir, "cohort_bad_class_percentiles.csv"),
            index=False,
        )

    _plot_germline_calibration(
        anchor,
        pooled,
        colmap,
        thr,
        os.path.join(out_dir, "cohort_threshold_calibration.png"),
    )

    print("\nWrote cohort_pooled.csv, cohort_recurrence.csv, "
          "cohort_per_sample_germline_counts.csv, "
          "cohort_practical_threshold_grid_search.csv, "
          "cohort_practical_anchor_percentiles.csv"
          + (", cohort_bad_class_percentiles.csv" if not bad_pct.empty else ""))

    return pooled, rec, anchor, thr, curve


# =============================================================================
# Notebook-facing germline/CNV marker workflow wrappers
# =============================================================================
# These helpers package the notebook chunks used for:
#   1. cohort-calibrated all-variant germline export
#   2. WT/clean-cell concordance and final germline seed selection
#   3. significant CNV/BAF marker audit against all_variant_germline_df
#   4. final no-loss CNV marker folding
#   5. AF clustering of variants
#   6. renamed AF-cluster germline update
#
# All functions are explicit and side-effect-light: they return updated objects
# rather than relying on notebook globals. Optional CSV/plot outputs are still
# available via save_dir / plot flags.


EMPIRICAL_COHORT_GERMLINE_THRESHOLDS = {
    # Clean canonical HET tier
    "clean_min_frac_het": 0.601,
    "clean_max_frac_hom": 0.182,
    "clean_max_frac_wt": 0.239,
    "clean_af_lower": 42.303,
    "clean_af_upper": 59.061,
    "clean_max_af_z": 12.275,

    # Backwards-compatible aliases used by apply_calibrated_germline_rules()
    "min_frac_het": 0.601,
    "max_frac_hom": 0.182,
    "max_frac_wt": 0.239,
    "af_lower": 42.303,
    "af_upper": 59.061,
    "max_af_z": 12.275,

    # Practical usable HET tier
    "usable_min_frac_het": 0.551,
    "usable_max_frac_hom": 0.200,
    "usable_max_frac_wt": 0.430,

    # Audit metrics from cohort calibration
    "usable_min_frac_het_percentile": 0.000,
    "usable_max_frac_hom_percentile": 100.000,
    "usable_max_frac_wt_percentile": 100.000,
    "usable_anchor_recall": 1.000,
    "usable_labelled_precision_vs_bad": 0.867,
    "usable_bad_passing_raw": 468,
    "usable_hard_excluded_passing_raw": 468,
    "usable_nonanchor_allowed_passing": 8,
    "usable_current_rejected": 0,
    "usable_current_retained": 3051,
    "usable_total_passing_after_hard_exclusions": 3064,
}


FINAL_NO_LOSS_CNV_MARKER_PARAMETERS = {
    "key_cols": ["sample_key_norm", "variant"],
    "final_cnv_informative_marker_no_loss": (
        "cnv_informative_het_rescue_loh_aware OR parameterised_cnv_informative_marker"
    ),
    "preserve_old_strict_rescue": "cnv_informative_het_rescue",
    "preserve_old_loh_aware_rescue": "cnv_informative_het_rescue_loh_aware",
    "preserve_old_loh_distorted_marker": "loh_distorted_wt_het_marker",
    "preserve_old_loh_reason": "loh_aware_rescue_reason",
    "parameterised_marker_classes": [
        "cnv_marker_het_centered_loh_distorted",
        "cnv_marker_fixed_like_loh_informative",
        "cnv_marker_amplification_informative",
    ],
    "p_max": 0.01,
    "baf_deviation_min": 0.30,
    "event_af_iqr_max": 0.20,
    "min_wt_valid": 5,
    "het_median_lower": 25,
    "het_median_upper": 75,
    "loh_relaxed_min_wt_frac_het_like": 0.15,
    "reflike_median_max": 5,
    "altlike_median_min": 95,
    "fixed_like_max_wt_frac_het_like": 0.05,
    "amp_depth_ratio_min": 1.30,
    "loh_like_events": ["LOH", "cnLOH", "DELETION"],
    "amplification_events": ["AMPLIFICATION"],
    "exclude_blacklisted": True,
    "exclude_pathogenic": True,
}


AF_CLUSTER_ONLY_DEFAULT_PARAMS = {
    "min_k": 2,
    "max_k": 20,
    "random_state": 7,
    "n_init": 50,
    "prefer_original_other_vars": True,
    "plot_heatmap_before_summary": True,
    "max_xticks": 60,
    "show_heatmap_variant_table": True,
    "show_cluster_color_key": True,
    "draw_cluster_boundaries": True,
    "show_cluster_names_on_top": True,
    "cluster_label_prefix": "Cluster",
    "min_cluster_label_width": 3,
}


DEFAULT_AF_CLUSTER_RENAME_MAP = {
    0: "germline_het",
    1: "germline_het",
    2: "germline_het",
    3: "germline_wt",
    4: "germline_hom",
}


def get_empirical_cohort_germline_thresholds(include_audit_metrics: bool = True):
    """Return a copy of the empirical cohort-calibrated thresholds."""
    thresholds = dict(EMPIRICAL_COHORT_GERMLINE_THRESHOLDS)
    if not include_audit_metrics:
        thresholds = {
            k: v for k, v in thresholds.items()
            if not k.startswith("usable_") or k in {
                "usable_min_frac_het",
                "usable_max_frac_hom",
                "usable_max_frac_wt",
            }
        }
    return thresholds


def _display(obj):
    """Notebook-friendly display without requiring IPython outside notebooks."""
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def _chr_pos_key(v):
    import re
    m = re.match(r"^(chr)?([^:]+):(\d+):", str(v))
    if not m:
        return (999, np.inf, str(v))
    chrom = m.group(2).replace("chr", "").upper()
    if chrom.isdigit():
        chrom_key = int(chrom)
    elif chrom == "X":
        chrom_key = 23
    elif chrom == "Y":
        chrom_key = 24
    elif chrom in {"M", "MT"}:
        chrom_key = 25
    else:
        chrom_key = 999
    return (chrom_key, int(m.group(3)), str(v))


def _variant_key_series(s):
    return (
        pd.Series(s, dtype=str)
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
    )


def _extract_variant_list(obj, name="variant_source"):
    """Extract variant strings from a list-like object or a variant dataframe."""
    if obj is None:
        return []
    if isinstance(obj, pd.DataFrame):
        for c in ["variant", "Variant", "id", "ID", "variant_id", "Variant_ID"]:
            if c in obj.columns:
                return _variant_key_series(obj[c]).dropna().drop_duplicates().tolist()
        raise KeyError(
            f"{name} is a DataFrame but no variant-like column was found. "
            f"Available columns: {list(obj.columns)}"
        )
    if isinstance(obj, pd.Series):
        return _variant_key_series(obj).dropna().drop_duplicates().tolist()
    if isinstance(obj, (list, tuple, set, np.ndarray, pd.Index)):
        return _variant_key_series(list(obj)).dropna().drop_duplicates().tolist()
    return _variant_key_series([obj]).dropna().drop_duplicates().tolist()


def _coerce_bool_like(series):
    if series is None:
        return None
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float).ne(0)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "t", "yes", "y", "1", "pass", "passed", "positive", "informative"])
    )


def _first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def run_empirical_all_variant_germline_export(
    Sample,
    *,
    experiment: str,
    patient_id: str,
    annotation_df,
    known_het_variants=None,
    DNA_Utils=None,
    keep_labels=("CD4_T", "CD8_T", "Other_T", "B", "Plasma"),
    out_dir: str = GERMLINE_PATTERN_DIR,
    thresholds=None,
    descriptive_min_dp: int = 2,
    placement_min_dp: int = 5,
    max_frac_wt_for_het_export: float = 0.90,
    verbose: bool = True,
    show_head: bool = False,
):
    """Run the first workflow chunk: export all-variant germline evidence and apply
    calibrated clean/practical HET tiers.

    Returns a dictionary containing ``all_variant_germline_df``,
    ``Sample_T_subset``, ``germline_profile_meta``, ``germline``,
    ``clean_germline``, ``rescued_germline``, and ``ref``.
    """
    thresholds = dict(thresholds or EMPIRICAL_COHORT_GERMLINE_THRESHOLDS)

    run_export = (
        DNA_Utils.run_all_variant_germline_profile_export
        if DNA_Utils is not None and hasattr(DNA_Utils, "run_all_variant_germline_profile_export")
        else run_all_variant_germline_profile_export
    )
    apply_rules = (
        DNA_Utils.apply_calibrated_germline_rules
        if DNA_Utils is not None and hasattr(DNA_Utils, "apply_calibrated_germline_rules")
        else apply_calibrated_germline_rules
    )

    all_variant_germline_df, Sample_T_subset, germline_profile_meta = run_export(
        Sample,
        experiment=experiment,
        patient_id=patient_id,
        annotation_df=annotation_df,
        known_het_variants=known_het_variants,
        DNA_Utils=DNA_Utils,
        keep_labels=list(keep_labels),
        out_dir=out_dir,
        descriptive_min_dp=descriptive_min_dp,
        placement_min_dp=placement_min_dp,
        min_frac_het_for_het=thresholds["usable_min_frac_het"],
        max_frac_hom_for_het=thresholds["usable_max_frac_hom"],
        max_frac_wt_for_het=max_frac_wt_for_het_export,
        verbose=verbose,
    )

    all_variant_germline_df = apply_rules(all_variant_germline_df, thresholds)

    germline = all_variant_germline_df.loc[
        all_variant_germline_df["calibrated_usable_het_seed"].fillna(False).astype(bool),
        "variant",
    ].astype(str).tolist()
    clean_germline = all_variant_germline_df.loc[
        all_variant_germline_df["calibrated_clean_het_seed"].fillna(False).astype(bool),
        "variant",
    ].astype(str).tolist()
    rescued_germline = all_variant_germline_df.loc[
        all_variant_germline_df["calibrated_rescued_het_seed"].fillna(False).astype(bool),
        "variant",
    ].astype(str).tolist()
    ref = germline_profile_meta["ref"]

    if verbose:
        print(f"Clean canonical germline HETs: {len(clean_germline)}")
        print(f"Rescued/skewed germline HETs:  {len(rescued_germline)}")
        print(f"Practical usable germline HETs: {len(germline)}")
    if show_head:
        _display(all_variant_germline_df.head(40))

    return {
        "thresholds": thresholds,
        "all_variant_germline_df": all_variant_germline_df,
        "Sample_T_subset": Sample_T_subset,
        "germline_profile_meta": germline_profile_meta,
        "germline": germline,
        "clean_germline": clean_germline,
        "rescued_germline": rescued_germline,
        "ref": ref,
    }


def add_wt_concordance_and_finalize_germline(
    Sample,
    all_variant_germline_df,
    whitelist,
    *,
    DNA_Utils=None,
    label: str = "wt",
    thresholds=None,
    max_ado_score: float = 0.8,
    min_clone_cells: int = 5,
    total_barcodes: int = None,
    max_frac_wt_for_het_export: float = 0.90,
    recipe: str = "calibrated_practical_or_confirmed",
    verbose: bool = True,
):
    """Run the second workflow chunk: build genotype-defined WT/clean-cell subset,
    add population concordance, reapply calibrated rules, and choose the final
    downstream ``germline_het_seed``.

    ``recipe`` options:
      - ``calibrated_practical_or_confirmed``: calibrated practical lymphoid OR
        concordant HET in WT population. This matches the user's notebook chunk.
      - ``calibrated_practical_and_confirmed``: stricter intersection.
      - ``calibrated_practical_or_wt_structure``: fallback for under-powered
        lymphoid subsets.
    """
    thresholds = dict(thresholds or EMPIRICAL_COHORT_GERMLINE_THRESHOLDS)

    build_wt = (
        DNA_Utils.build_wt_cell_subset
        if DNA_Utils is not None and hasattr(DNA_Utils, "build_wt_cell_subset")
        else build_wt_cell_subset
    )
    add_conc = (
        DNA_Utils.add_population_concordance
        if DNA_Utils is not None and hasattr(DNA_Utils, "add_population_concordance")
        else add_population_concordance
    )
    apply_rules = (
        DNA_Utils.apply_calibrated_germline_rules
        if DNA_Utils is not None and hasattr(DNA_Utils, "apply_calibrated_germline_rules")
        else apply_calibrated_germline_rules
    )

    df0 = all_variant_germline_df.copy()
    stale = [
        c for c in df0.columns
        if c.startswith(f"{label}_")
        or c in {
            "concordant_germline_het",
            "germline_het_seed",
            "calibrated_clean_het_seed",
            "calibrated_rescued_het_seed",
            "calibrated_usable_het_seed",
            "calibrated_rejected_het_seed",
            "calibrated_reject_reasons",
        }
    ]
    df0 = df0.drop(columns=stale, errors="ignore")

    if total_barcodes is None:
        total_barcodes = len(Sample.dna.row_attrs["barcode"])

    _build_wt_result = build_wt(
        Sample,
        whitelist,
        DNA_Utils=DNA_Utils,
        max_ado_score=max_ado_score,
        min_clone_cells=min_clone_cells,
        total_barcodes=total_barcodes,
        verbose=verbose,
    )

    # Backward-compatible unpacking. Older build_wt_cell_subset returns
    # (Sample_wt, clone_df); future/local variants may return additional
    # barcode-level clone objects. Prefer explicit returns when present, else
    # read the attrs attached to clone_df by this implementation.
    if isinstance(_build_wt_result, tuple) and len(_build_wt_result) >= 2:
        Sample_wt, clone_df = _build_wt_result[:2]
        clone_barcode_df = (
            _build_wt_result[2]
            if len(_build_wt_result) >= 3
            else clone_df.attrs.get("clone_barcode_df", pd.DataFrame())
        )
        cell_clone_labels = (
            _build_wt_result[3]
            if len(_build_wt_result) >= 4
            else clone_df.attrs.get("cell_clone_labels", pd.Series(dtype=object, name="clone_label"))
        )
        step2_clone_labels_used = (
            _build_wt_result[4]
            if len(_build_wt_result) >= 5
            else clone_df.attrs.get("step2_clone_labels_used", _get_clone_columns_from_clone_df(clone_df))
        )
    else:
        raise RuntimeError("build_wt_cell_subset returned an unexpected object.")

    if not isinstance(clone_barcode_df, pd.DataFrame):
        clone_barcode_df = pd.DataFrame(clone_barcode_df)
    if not isinstance(cell_clone_labels, pd.Series):
        if isinstance(clone_barcode_df, pd.DataFrame) and {"barcode", "clone_label"}.issubset(clone_barcode_df.columns):
            cell_clone_labels = pd.Series(
                clone_barcode_df["clone_label"].astype(str).values,
                index=clone_barcode_df["barcode"].astype(str).values,
                name="clone_label",
            )
        else:
            cell_clone_labels = pd.Series(dtype=object, name="clone_label")

    if verbose and isinstance(clone_barcode_df, pd.DataFrame) and not clone_barcode_df.empty:
        print("  Step2 clone barcode membership:")
        print(clone_barcode_df["clone_label"].astype(str).value_counts(dropna=False).to_string())

    df = add_conc(
        df0,
        Sample_wt,
        label=label,
        min_frac_het_for_het=thresholds["usable_min_frac_het"],
        max_frac_hom_for_het=thresholds["usable_max_frac_hom"],
        max_frac_wt_for_het=max_frac_wt_for_het_export,
        verbose=verbose,
    )

    df = apply_rules(df, thresholds)

    calibrated_practical = df["calibrated_usable_het_seed"].fillna(False).astype(bool)
    calibrated_clean = df["calibrated_clean_het_seed"].fillna(False).astype(bool)
    calibrated_rescued = df["calibrated_rescued_het_seed"].fillna(False).astype(bool)

    lymphoid = df.get("usable_germline_het_seed", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    confirmed = df.get("concordant_germline_het", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    wt_het_col = f"{label}_germline_het_seed"
    wt_het = df.get(wt_het_col, pd.Series(False, index=df.index)).fillna(False).astype(bool)

    wt_frac_het = pd.to_numeric(df.get(f"{label}_frac_HET"), errors="coerce")
    wt_frac_hom = pd.to_numeric(df.get(f"{label}_frac_HOM"), errors="coerce")
    wt_frac_wt = pd.to_numeric(df.get(f"{label}_frac_WT"), errors="coerce")

    wt_structure_supported = (
        wt_het
        & (wt_frac_het >= thresholds["usable_min_frac_het"])
        & (wt_frac_hom <= thresholds["usable_max_frac_hom"])
        & (wt_frac_wt <= thresholds["usable_max_frac_wt"])
    )

    if recipe == "calibrated_practical_or_confirmed":
        final_mask = calibrated_practical | confirmed
    elif recipe == "calibrated_practical_and_confirmed":
        final_mask = calibrated_practical & confirmed
    elif recipe == "calibrated_practical_or_wt_structure":
        final_mask = calibrated_practical | wt_structure_supported
    else:
        raise ValueError(
            "recipe must be one of: calibrated_practical_or_confirmed, "
            "calibrated_practical_and_confirmed, calibrated_practical_or_wt_structure"
        )

    df["germline_het_seed"] = final_mask.fillna(False).astype(bool)

    germline = df.loc[df["germline_het_seed"], "variant"].astype(str).tolist()
    clean_germline = df.loc[calibrated_clean, "variant"].astype(str).tolist()
    rescued_germline = df.loc[calibrated_rescued, "variant"].astype(str).tolist()

    if verbose:
        print(f"lymphoid usable, original:       {int(lymphoid.sum())}")
        print(f"calibrated clean canonical:      {int(calibrated_clean.sum())}")
        print(f"calibrated rescued/skewed:       {int(calibrated_rescued.sum())}")
        print(f"calibrated practical usable:     {int(calibrated_practical.sum())}")
        print(f"confirmed HET in WT + lymphoid:  {int(confirmed.sum())}")
        print(f"WT HET by donor-genotype call:   {int(wt_het.sum())}")
        print(f"WT structure-supported HET:      {int(wt_structure_supported.sum())}")
        print(f"germline downstream final:       {len(germline)}")

    return {
        "all_variant_germline_df": df,
        "Sample_wt": Sample_wt,
        "clone_df": clone_df,
        "clone_barcode_df": clone_barcode_df,
        "cell_clone_labels": cell_clone_labels,
        "step2_clone_labels_used": step2_clone_labels_used,
        "germline": germline,
        "clean_germline": clean_germline,
        "rescued_germline": rescued_germline,
        "calibrated_practical": calibrated_practical,
        "calibrated_clean": calibrated_clean,
        "calibrated_rescued": calibrated_rescued,
        "lymphoid": lymphoid,
        "confirmed": confirmed,
        "wt_het": wt_het,
        "wt_structure_supported": wt_structure_supported,
        "recipe": recipe,
    }


def _build_significant_variant_summary(
    *,
    significant_variant_summary=None,
    sig_res=None,
    sig_variants_ordered=None,
    variant_col=None,
    clone_col=None,
    event_col=None,
    p_col=None,
    baf_col=None,
    iqr_col=None,
    depth_col=None,
):
    """Build/standardise the significant-variant summary used by germline/CNV audits."""
    if significant_variant_summary is not None:
        out = significant_variant_summary.copy()
        if "variant" not in out.columns:
            raise KeyError("significant_variant_summary must contain a 'variant' column.")
        out["variant"] = out["variant"].astype(str)
        out["significant_informative"] = True
    else:
        if sig_res is None:
            raise RuntimeError(
                "Provide significant_variant_summary or sig_res from the heatmap/significant extraction block."
            )
        if variant_col is None:
            variant_col = _first_existing_col(
                sig_res,
                ["variant", "variant_id", "id", "Variant", "Variant ID", "mutation", "locus"],
            )
        if variant_col is None:
            raise ValueError(f"Could not identify variant column in sig_res: {sig_res.columns.tolist()}")
        tmp = sig_res.copy()
        tmp["_variant"] = tmp[variant_col].astype(str)
        agg_dict = {"n_informative_comparisons": ("_variant", "size")}
        if "chr_arm" in tmp.columns:
            agg_dict["informative_chr_arms"] = (
                "chr_arm", lambda x: ",".join(sorted(set(map(str, x))))
            )
        if clone_col is not None and clone_col in tmp.columns:
            agg_dict["informative_clone_labels"] = (
                clone_col, lambda x: ",".join(sorted(set(map(str, x))))
            )
        if event_col is not None and event_col in tmp.columns:
            agg_dict["informative_event_types"] = (
                event_col, lambda x: ",".join(sorted(set(map(str, x))))
            )
        if p_col is not None and p_col in tmp.columns:
            tmp["_p"] = pd.to_numeric(tmp[p_col], errors="coerce")
            agg_dict["min_p"] = ("_p", "min")
        if baf_col is not None and baf_col in tmp.columns:
            tmp["_abs_baf"] = pd.to_numeric(tmp[baf_col], errors="coerce").abs()
            agg_dict["max_abs_baf_deviation"] = ("_abs_baf", "max")
        if iqr_col is not None and iqr_col in tmp.columns:
            tmp["_iqr"] = pd.to_numeric(tmp[iqr_col], errors="coerce")
            agg_dict["max_af_iqr"] = ("_iqr", "max")
        if depth_col is not None and depth_col in tmp.columns:
            tmp["_depth"] = pd.to_numeric(tmp[depth_col], errors="coerce")
            agg_dict["min_depth_ratio"] = ("_depth", "min")
            agg_dict["max_depth_ratio"] = ("_depth", "max")
        out = (
            tmp.groupby("_variant")
            .agg(**agg_dict)
            .reset_index()
            .rename(columns={"_variant": "variant"})
        )
        out["variant"] = out["variant"].astype(str)
        out["significant_informative"] = True

    if sig_variants_ordered is not None:
        cats = [str(v) for v in sig_variants_ordered]
        out["variant"] = pd.Categorical(out["variant"], categories=cats, ordered=True)
        out = out.sort_values("variant").reset_index(drop=True)
        out["variant"] = out["variant"].astype(str)
    else:
        out = out.sort_values("variant", key=lambda s: s.map(_chr_pos_key)).reset_index(drop=True)
    return out


def check_significant_variants_against_all_variant_germline_df(
    *,
    all_variant_germline_df,
    df_AF_all,
    df_DP_all=None,
    experiment: str = None,
    patient_id: str = None,
    significant_variant_summary=None,
    sig_res=None,
    sig_variants_ordered=None,
    variant_col=None,
    clone_col=None,
    event_col=None,
    p_col=None,
    baf_col=None,
    iqr_col=None,
    depth_col=None,
    save_dir: str = GERMLINE_PATTERN_DIR,
    wt_labels=("WT",),
    min_wt_cells: int = 5,
    min_wt_valid_cells: int = 5,
    min_wt_dp: int = 5,
    wt_het_af_lower: float = 25,
    wt_het_af_upper: float = 75,
    min_wt_frac_het_like: float = 0.40,
    max_wt_af_iqr: float = 35,
    allow_blacklisted_as_cnv_markers: bool = False,
    allow_pathogenic_as_cnv_markers: bool = False,
    save_outputs: bool = True,
    verbose: bool = True,
):
    """Check significant/informative variants directly against the current
    ``all_variant_germline_df``.

    This is the post-heatmap audit function that produces:
      - ``audit_real``: every significant variant checked against the table.
      - ``missed_significant_in_all_variant_germline_df``: significant variants
        present in ``all_variant_germline_df`` but not in the downstream seed.
      - ``real_missed_variants_df``: stricter WT-HET-supported rescue subset.
    """
    if "variant" not in all_variant_germline_df.columns:
        raise KeyError("all_variant_germline_df must contain a 'variant' column.")
    if "label" not in df_AF_all.columns:
        raise KeyError("df_AF_all must contain a 'label' column.")

    germline_audit_df = all_variant_germline_df.copy()
    germline_audit_df["variant"] = germline_audit_df["variant"].astype(str)
    dup_variants = int(germline_audit_df["variant"].duplicated().sum())
    if dup_variants > 0:
        if verbose:
            print(f"WARNING: all_variant_germline_df has {dup_variants} duplicate variant rows.")
            print("Keeping the first row per variant for this audit.")
        germline_audit_df = germline_audit_df.drop_duplicates("variant", keep="first")

    significant_variant_summary = _build_significant_variant_summary(
        significant_variant_summary=significant_variant_summary,
        sig_res=sig_res,
        sig_variants_ordered=sig_variants_ordered,
        variant_col=variant_col,
        clone_col=clone_col,
        event_col=event_col,
        p_col=p_col,
        baf_col=baf_col,
        iqr_col=iqr_col,
        depth_col=depth_col,
    )

    audit_real = significant_variant_summary.merge(
        germline_audit_df,
        on="variant",
        how="left",
        indicator="in_all_variant_germline_df",
    )
    audit_real["in_all_variant_germline_df"] = audit_real["in_all_variant_germline_df"].eq("both")

    bool_like_cols = [
        "germline_het_seed",
        "calibrated_usable_het_seed",
        "calibrated_clean_het_seed",
        "calibrated_rescued_het_seed",
        "usable_germline_het_seed",
        "clean_germline_het_seed",
        "rescued_germline_het_seed",
        "concordant_germline_het",
        "is_blacklisted",
        "blacklisted",
        "is_pathogenic_excluded",
        "pathogenic_excluded",
    ]
    for c in bool_like_cols:
        if c in audit_real.columns:
            audit_real[c] = _coerce_bool_like(audit_real[c]).fillna(False).astype(bool)

    downstream_seed_col = None
    for c in [
        "germline_het_seed",
        "calibrated_usable_het_seed",
        "usable_germline_het_seed",
    ]:
        if c in audit_real.columns:
            downstream_seed_col = c
            break
    if downstream_seed_col is None:
        audit_real["downstream_seeded"] = False
    else:
        audit_real["downstream_seeded"] = audit_real[downstream_seed_col].fillna(False).astype(bool)

    audit_real["missed_by_downstream_seed"] = (
        audit_real["significant_informative"].fillna(False).astype(bool)
        & audit_real["in_all_variant_germline_df"].fillna(False).astype(bool)
        & ~audit_real["downstream_seeded"].fillna(False).astype(bool)
    )

    labels = df_AF_all["label"].astype(str)
    wt_mask = labels.isin(list(wt_labels))
    if int(wt_mask.sum()) < min_wt_cells:
        raise RuntimeError(
            f"Only {int(wt_mask.sum())} WT/reference cells found. Need at least {min_wt_cells}."
        )

    wt_rows = []
    for v in audit_real["variant"].astype(str).drop_duplicates():
        if v not in df_AF_all.columns:
            wt_rows.append({
                "variant": v,
                "wt_n_cells": int(wt_mask.sum()),
                "wt_n_valid": 0,
                "wt_median_af": np.nan,
                "wt_af_iqr": np.nan,
                "wt_frac_het_like": np.nan,
                "wt_frac_ref_like": np.nan,
                "wt_frac_alt_like": np.nan,
                "wt_het_supported": False,
            })
            continue
        af = pd.to_numeric(df_AF_all.loc[wt_mask, v], errors="coerce")
        if df_DP_all is not None and v in df_DP_all.columns:
            dp = pd.to_numeric(df_DP_all.loc[wt_mask, v], errors="coerce")
            valid = af.notna() & dp.notna() & (dp >= min_wt_dp)
        else:
            valid = af.notna()
        af_valid = af.loc[valid]
        n_valid = int(af_valid.shape[0])
        if n_valid > 0:
            wt_median_af = float(af_valid.median())
            wt_af_iqr = float(af_valid.quantile(0.75) - af_valid.quantile(0.25))
            wt_frac_het_like = float(af_valid.between(wt_het_af_lower, wt_het_af_upper).mean())
            wt_frac_ref_like = float((af_valid <= 5).mean())
            wt_frac_alt_like = float((af_valid >= 95).mean())
        else:
            wt_median_af = np.nan
            wt_af_iqr = np.nan
            wt_frac_het_like = np.nan
            wt_frac_ref_like = np.nan
            wt_frac_alt_like = np.nan
        wt_het_supported = (
            n_valid >= min_wt_valid_cells
            and pd.notna(wt_median_af)
            and wt_het_af_lower <= wt_median_af <= wt_het_af_upper
            and pd.notna(wt_af_iqr)
            and wt_af_iqr <= max_wt_af_iqr
            and pd.notna(wt_frac_het_like)
            and wt_frac_het_like >= min_wt_frac_het_like
        )
        wt_rows.append({
            "variant": v,
            "wt_n_cells": int(wt_mask.sum()),
            "wt_n_valid": n_valid,
            "wt_median_af": wt_median_af,
            "wt_af_iqr": wt_af_iqr,
            "wt_frac_het_like": wt_frac_het_like,
            "wt_frac_ref_like": wt_frac_ref_like,
            "wt_frac_alt_like": wt_frac_alt_like,
            "wt_het_supported": bool(wt_het_supported),
        })
    audit_real = audit_real.merge(pd.DataFrame(wt_rows), on="variant", how="left")

    def _flag_any(row, cols):
        return any(c in row.index and bool(row.get(c, False)) for c in cols)

    def _str_val(row, col):
        return str(row[col]) if col in row.index and pd.notna(row[col]) else ""

    audit_real["excluded_blacklisted_any"] = audit_real.apply(
        lambda r: _flag_any(r, ["is_blacklisted", "blacklisted"]), axis=1
    )
    audit_real["excluded_pathogenic_any"] = audit_real.apply(
        lambda r: _flag_any(r, ["is_pathogenic_excluded", "pathogenic_excluded"]), axis=1
    )

    blacklist_ok = (
        ~audit_real["excluded_blacklisted_any"]
        if not allow_blacklisted_as_cnv_markers
        else pd.Series(True, index=audit_real.index)
    )
    pathogenic_ok = (
        ~audit_real["excluded_pathogenic_any"]
        if not allow_pathogenic_as_cnv_markers
        else pd.Series(True, index=audit_real.index)
    )

    def _miss_reason(row):
        if not bool(row.get("significant_informative", False)):
            return "not_significant"
        if not bool(row.get("in_all_variant_germline_df", False)):
            return "not_present_in_all_variant_germline_df"
        if bool(row.get("downstream_seeded", False)):
            return "kept_downstream"
        if bool(row.get("excluded_blacklisted_any", False)):
            return "excluded_blacklisted"
        if bool(row.get("excluded_pathogenic_any", False)):
            return "excluded_pathogenic"
        donor_genotype = _str_val(row, "donor_genotype")
        if donor_genotype and donor_genotype.lower() not in {"het", "heterozygous", "nan"}:
            return f"donor_genotype_not_het__{donor_genotype}"
        broad = _str_val(row, "germline_broad_class")
        if broad:
            return f"germline_broad_class__{broad}"
        status = _str_val(row, "status")
        review = _str_val(row, "review_verdict")
        if status and review:
            return f"status__{status}__review__{review}"
        if status:
            return f"status__{status}"
        if review:
            return f"review__{review}"
        return "present_but_not_seeded_unknown"

    audit_real["miss_reason"] = audit_real.apply(_miss_reason, axis=1)
    audit_real["cnv_informative_het_seed"] = (
        audit_real["significant_informative"].fillna(False).astype(bool)
        & audit_real["in_all_variant_germline_df"].fillna(False).astype(bool)
        & audit_real["wt_het_supported"].fillna(False).astype(bool)
        & blacklist_ok
        & pathogenic_ok
    )
    audit_real["cnv_informative_het_rescue"] = (
        audit_real["cnv_informative_het_seed"].fillna(False).astype(bool)
        & audit_real["missed_by_downstream_seed"].fillna(False).astype(bool)
    )

    def _cnv_marker_reason(row):
        if not bool(row.get("significant_informative", False)):
            return "not_significant"
        if not bool(row.get("in_all_variant_germline_df", False)):
            return "not_in_all_variant_germline_df"
        if not bool(row.get("wt_het_supported", False)):
            return "not_wt_het_supported"
        if bool(row.get("excluded_blacklisted_any", False)) and not allow_blacklisted_as_cnv_markers:
            return "excluded_blacklisted"
        if bool(row.get("excluded_pathogenic_any", False)) and not allow_pathogenic_as_cnv_markers:
            return "excluded_pathogenic"
        if bool(row.get("downstream_seeded", False)):
            return "already_kept_by_downstream_seed"
        donor_genotype = _str_val(row, "donor_genotype")
        if donor_genotype:
            return f"rescued_significant_wt_het__donor_called_{donor_genotype}"
        return "rescued_significant_wt_het"

    audit_real["cnv_marker_reason"] = audit_real.apply(_cnv_marker_reason, axis=1)

    missed_significant_in_all_variant_germline_df = audit_real.loc[
        audit_real["missed_by_downstream_seed"].fillna(False).astype(bool)
    ].copy()
    real_missed_variants_df = audit_real.loc[
        audit_real["cnv_informative_het_rescue"].fillna(False).astype(bool)
    ].copy()

    priority_cols = [
        "variant", "cnv_informative_het_seed", "cnv_informative_het_rescue",
        "cnv_marker_reason", "miss_reason", "significant_informative",
        "downstream_seeded", "missed_by_downstream_seed", "in_all_variant_germline_df",
        downstream_seed_col, "donor_genotype", "germline_broad_class", "status", "review_verdict",
        "wt_n_valid", "wt_median_af", "wt_af_iqr", "wt_frac_het_like", "wt_frac_ref_like",
        "wt_frac_alt_like", "n_informative_comparisons", "informative_chr_arms",
        "informative_clone_labels", "informative_event_types", "min_p", "max_abs_baf_deviation",
        "max_af_iqr", "min_depth_ratio", "max_depth_ratio", "excluded_blacklisted_any",
        "excluded_pathogenic_any", "is_blacklisted", "is_pathogenic_excluded",
    ]
    priority_cols = [c for c in priority_cols if c is not None and c in audit_real.columns]
    ordered_cols = priority_cols + [c for c in audit_real.columns if c not in priority_cols]
    audit_real = audit_real[ordered_cols].copy()
    missed_significant_in_all_variant_germline_df = missed_significant_in_all_variant_germline_df[
        [c for c in ordered_cols if c in missed_significant_in_all_variant_germline_df.columns]
    ].copy()
    real_missed_variants_df = real_missed_variants_df[
        [c for c in ordered_cols if c in real_missed_variants_df.columns]
    ].copy()

    sort_cols = [c for c in ["min_p", "variant"] if c in audit_real.columns]
    if sort_cols:
        audit_real = audit_real.sort_values(sort_cols).reset_index(drop=True)
        missed_significant_in_all_variant_germline_df = missed_significant_in_all_variant_germline_df.sort_values(sort_cols).reset_index(drop=True)
        real_missed_variants_df = real_missed_variants_df.sort_values(sort_cols).reset_index(drop=True)

    paths = {}
    if save_outputs:
        if experiment is None or patient_id is None:
            raise ValueError("experiment and patient_id are required when save_outputs=True.")
        os.makedirs(save_dir, exist_ok=True)
        paths["audit_real_path"] = os.path.join(
            save_dir,
            f"{experiment}_{patient_id}_significant_variants_checked_against_all_variant_germline_df.csv",
        )
        paths["missed_in_all_germline_path"] = os.path.join(
            save_dir,
            f"{experiment}_{patient_id}_missed_significant_in_all_variant_germline_df.csv",
        )
        paths["real_missed_path"] = os.path.join(
            save_dir,
            f"{experiment}_{patient_id}_real_missed_variants.csv",
        )
        audit_real.to_csv(paths["audit_real_path"], index=False)
        missed_significant_in_all_variant_germline_df.to_csv(
            paths["missed_in_all_germline_path"], index=False
        )
        real_missed_variants_df.to_csv(paths["real_missed_path"], index=False)

    if verbose:
        print("\nSignificant variants checked against all_variant_germline_df")
        print("----------------------------------------------------------")
        print(f"Significant/informative variants:              {audit_real['variant'].nunique()}")
        print(f"Present in all_variant_germline_df:            {int(audit_real['in_all_variant_germline_df'].sum())}")
        print(f"Not present in all_variant_germline_df:        {int((~audit_real['in_all_variant_germline_df']).sum())}")
        print(f"Already kept by downstream seed:               {int(audit_real['downstream_seeded'].sum())}")
        print(f"Missed but present in all_variant_germline_df:  {missed_significant_in_all_variant_germline_df.shape[0]}")
        print(f"WT-HET-supported real rescued variants:        {real_missed_variants_df.shape[0]}")
        print(f"Downstream seed column used:                   {downstream_seed_col}")
        if not missed_significant_in_all_variant_germline_df.empty:
            print("\nMiss reasons among variants present in all_variant_germline_df but missed:")
            print(missed_significant_in_all_variant_germline_df["miss_reason"].value_counts(dropna=False).to_string())
        if save_outputs:
            print("\nSaved:")
            for p in paths.values():
                print(p)

    return {
        "audit_real": audit_real,
        "missed_significant_in_all_variant_germline_df": missed_significant_in_all_variant_germline_df,
        "real_missed_variants_df": real_missed_variants_df,
        "downstream_seed_col": downstream_seed_col,
        "paths": paths,
    }


def apply_final_no_loss_cnv_marker_rules(
    all_variant_germline_df,
    audit_real,
    *,
    parameters=None,
    verbose: bool = True,
):
    """Apply the final no-loss CNV marker rule and fold the final flag back
    into ``all_variant_germline_df``.

    Returns a dictionary containing the updated dataframe, updated ``audit_real``,
    ``filtered_germline_variants``, and ``germline_df_for_plot``.
    """
    P = dict(FINAL_NO_LOSS_CNV_MARKER_PARAMETERS)
    if parameters is not None:
        P.update(parameters)

    a = audit_real.copy()
    df = all_variant_germline_df.copy()
    if "variant" not in a.columns or "variant" not in df.columns:
        raise KeyError("Both audit_real and all_variant_germline_df must contain 'variant'.")

    num = lambda c: pd.to_numeric(a[c], errors="coerce") if c in a.columns else pd.Series(np.nan, index=a.index)
    def _bool(c):
        return _coerce_bool_like(a[c]).fillna(False).astype(bool) if c in a.columns else pd.Series(False, index=a.index)

    event = a.get("informative_event_types", pd.Series("", index=a.index)).fillna("").astype(str)
    cnv_event = (
        (num("min_p") <= P["p_max"])
        & (num("max_abs_baf_deviation") >= P["baf_deviation_min"])
        & (num("max_af_iqr") <= P["event_af_iqr_max"])
    )
    loh_event = event.str.contains("|".join(P["loh_like_events"]), case=False, regex=True)
    amp_event = event.str.contains("|".join(P["amplification_events"]), case=False, regex=True)

    wt_valid = num("wt_n_valid") >= P["min_wt_valid"]
    wt_med = num("wt_median_af")
    wt_hl = num("wt_frac_het_like")
    wt_het_centred = (
        wt_valid
        & wt_med.between(P["het_median_lower"], P["het_median_upper"])
        & (wt_hl >= P["loh_relaxed_min_wt_frac_het_like"])
    )
    wt_fixed = (
        wt_valid
        & ((wt_med <= P["reflike_median_max"]) | (wt_med >= P["altlike_median_min"]))
        & (wt_hl <= P["fixed_like_max_wt_frac_het_like"])
    )

    excl = pd.Series(False, index=a.index)
    if P["exclude_pathogenic"]:
        excl |= _bool("is_pathogenic_excluded") | _bool("excluded_pathogenic_any")
    if P["exclude_blacklisted"]:
        excl |= _bool("is_blacklisted") | _bool("excluded_blacklisted_any")

    a["cnv_marker_het_centered_loh_distorted"] = wt_het_centred & cnv_event & loh_event & ~excl
    a["cnv_marker_fixed_like_loh_informative"] = wt_fixed & cnv_event & loh_event & ~excl
    a["cnv_marker_amplification_informative"] = (
        (num("max_depth_ratio") >= P["amp_depth_ratio_min"])
        & amp_event
        & cnv_event
        & ~excl
    )
    a["parameterised_cnv_informative_marker"] = (
        a["cnv_marker_het_centered_loh_distorted"]
        | a["cnv_marker_fixed_like_loh_informative"]
        | a["cnv_marker_amplification_informative"]
    )

    loh_aware = _bool("cnv_informative_het_rescue_loh_aware") | _bool("cnv_informative_het_rescue")
    a["final_cnv_informative_marker_no_loss"] = loh_aware | a["parameterised_cnv_informative_marker"]

    df = df.drop(columns=["final_cnv_informative_marker_no_loss", "germline_tier"], errors="ignore")
    fold = a[["variant", "final_cnv_informative_marker_no_loss"]].drop_duplicates("variant")
    df = df.merge(fold, on="variant", how="left")
    df["final_cnv_informative_marker_no_loss"] = df["final_cnv_informative_marker_no_loss"].fillna(False).astype(bool)

    clean_mask = df.get("germline_het_seed", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    cnv_mask = df["final_cnv_informative_marker_no_loss"].fillna(False).astype(bool)
    germline_mask = clean_mask | cnv_mask
    df["germline_tier"] = np.select(
        [clean_mask, cnv_mask],
        ["clean_het", "cnv_informative"],
        default="none",
    )

    filtered_germline_variants = df.loc[germline_mask, "variant"].astype(str).tolist()
    germline_df_for_plot = df.loc[germline_mask].copy()

    if verbose:
        print("\nFinal no-loss CNV marker fold-in")
        print("--------------------------------")
        print(f"clean germline_het_seed:                  {int(clean_mask.sum())}")
        print(f"final_cnv_informative_marker_no_loss:     {int(cnv_mask.sum())}")
        print(f"combined filtered_germline_variants:      {len(filtered_germline_variants)}")
        for c in P["parameterised_marker_classes"] + ["parameterised_cnv_informative_marker"]:
            if c in a.columns:
                print(f"{c:42s} {int(a[c].sum())}")

    return {
        "all_variant_germline_df": df,
        "audit_real": a,
        "filtered_germline_variants": filtered_germline_variants,
        "germline_df_for_plot": germline_df_for_plot,
        "parameters": P,
    }


def run_af_clustering_only(
    *,
    df_AF_all,
    filtered_germline_variants,
    original_other_vars_for_cluster=None,
    other_vars=None,
    patient_id: str = None,
    params=None,
    plot: bool = True,
    show_tables: bool = True,
):
    """Run AF clustering only: variant -> AF cluster plus germline-count summaries.

    This wraps the user's second script without applying any rescue or updating
    ``all_variant_germline_df``.

    In addition to the per-clone label medians, a pooled WT-vs-non-WT contrast
    is included as a feature so LOH-distorted variants (het in WT, fixed in the
    dominant clone) separate onto their own axis regardless of how many clone
    labels step3 emits.
    """
    import os
    import random
    import re
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_hex
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    P = dict(AF_CLUSTER_ONLY_DEFAULT_PARAMS)
    if params is not None:
        P.update(params)

    seed = int(P["random_state"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if "label" in df_AF_all.columns:
        af_cols = [c for c in df_AF_all.columns if c != "label"]
        labels = df_AF_all["label"].astype(str)
    else:
        af_cols = list(df_AF_all.columns)
        labels = pd.Series("all_cells", index=df_AF_all.index)

    af_mat = df_AF_all[af_cols].copy()
    af_mat.columns = _variant_key_series(pd.Series(af_mat.columns)).values
    af_mat = af_mat.apply(pd.to_numeric, errors="coerce")
    all_af_vars = sorted(af_mat.columns.astype(str).tolist(), key=_chr_pos_key)
    n_cells_total = af_mat.shape[0]

    label_order = labels.value_counts().index.tolist()
    if "WT" in label_order:
        label_order = ["WT"] + [x for x in label_order if x != "WT"]

    filtered_germline_set = set(
        v for v in _extract_variant_list(filtered_germline_variants, "filtered_germline_variants")
        if v in af_mat.columns
    )
    filtered_germline_vars_for_cluster = sorted(list(filtered_germline_set), key=_chr_pos_key)

    if P["prefer_original_other_vars"] and original_other_vars_for_cluster is not None:
        other_vars_for_cluster = _extract_variant_list(original_other_vars_for_cluster, "original_other_vars_for_cluster")
        other_source_name = "original_other_vars_for_cluster"
    elif other_vars is not None and len(_extract_variant_list(other_vars, "other_vars")) > 0:
        other_vars_for_cluster = _extract_variant_list(other_vars, "other_vars")
        other_source_name = "other_vars"
    else:
        other_vars_for_cluster = [v for v in all_af_vars if v not in filtered_germline_set]
        other_source_name = "all_af_vars_minus_filtered_germline"

    other_vars_for_cluster = sorted(
        [v for v in other_vars_for_cluster if v in af_mat.columns and v not in filtered_germline_set],
        key=_chr_pos_key,
    )
    cluster_variants = sorted(
        pd.Series(filtered_germline_vars_for_cluster + other_vars_for_cluster, dtype=str)
        .drop_duplicates()
        .tolist(),
        key=_chr_pos_key,
    )

    print("\nAF clustering input")
    print("-------------------")
    print(f"AF matrix cells:                       {n_cells_total}")
    print(f"AF matrix variants:                    {len(all_af_vars)}")
    print(f"filtered_germline_variants in AF:      {len(filtered_germline_vars_for_cluster)}")
    print(f"other/candidate variant source:        {other_source_name}")
    print(f"other/candidate variants in AF:        {len(other_vars_for_cluster)}")
    print(f"total variants clustered:              {len(cluster_variants)}")
    print(f"AF clustering random seed:             {seed}")

    if len(cluster_variants) < 4:
        raise ValueError("Too few variants to cluster.")

    def _safe_iqr(x):
        x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        return np.nan if x.empty else float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

    def _af_feature_row(v):
        a = pd.to_numeric(af_mat[v], errors="coerce")
        valid = a.notna()
        n_valid = int(valid.sum())
        if n_valid == 0:
            return {
                "variant": v, "n_valid": 0, "valid_frac": 0.0,
                "median_af": np.nan, "mean_af": np.nan, "af_iqr": np.nan, "af_std": np.nan,
                "frac_het_like": np.nan, "frac_clean_het": np.nan,
                "frac_ref_like": np.nan, "frac_alt_like": np.nan, "frac_fixed_like": np.nan,
            }
        av = a.loc[valid]
        frac_ref_like = float((av <= 5).mean())
        frac_alt_like = float((av >= 95).mean())
        return {
            "variant": v,
            "n_valid": n_valid,
            "valid_frac": float(n_valid / n_cells_total),
            "median_af": float(np.nanmedian(av)),
            "mean_af": float(np.nanmean(av)),
            "af_iqr": _safe_iqr(av),
            "af_std": float(np.nanstd(av)),
            "frac_het_like": float(((av >= 25) & (av <= 75)).mean()),
            "frac_clean_het": float(((av >= 35) & (av <= 65)).mean()),
            "frac_ref_like": frac_ref_like,
            "frac_alt_like": frac_alt_like,
            "frac_fixed_like": float(frac_ref_like + frac_alt_like),
        }

    feature_df = pd.DataFrame([_af_feature_row(v) for v in cluster_variants])
    feature_df["is_filtered_germline_variant"] = feature_df["variant"].isin(filtered_germline_set)
    feature_df["is_other_or_candidate_variant"] = feature_df["variant"].isin(set(other_vars_for_cluster))

    label_rows = []
    for v in cluster_variants:
        row = {"variant": v}
        label_medians = []
        for lab in label_order:
            idx = labels[labels.eq(lab)].index
            vals = pd.to_numeric(af_mat.loc[idx, v], errors="coerce").dropna()
            safe_lab = re.sub(r"[^A-Za-z0-9_]+", "_", str(lab))
            col = f"label_median__{safe_lab}"
            if vals.empty:
                row[col] = np.nan
            else:
                med = float(np.nanmedian(vals))
                row[col] = med
                label_medians.append(med)

        # Pooled WT-vs-non-WT contrast alongside the per-clone medians: restores
        # the WT(~50)-vs-clone(~0/100) gap that identifies LOH-distorted variants,
        # independent of how many clone labels step3 emits.
        wt_idx = labels[labels.eq("WT")].index
        nonwt_idx = labels[labels.ne("WT")].index
        wt_vals = pd.to_numeric(af_mat.loc[wt_idx, v], errors="coerce").dropna()
        nonwt_vals = pd.to_numeric(af_mat.loc[nonwt_idx, v], errors="coerce").dropna()
        wt_med_pooled = float(np.nanmedian(wt_vals)) if len(wt_vals) else np.nan
        nonwt_med_pooled = float(np.nanmedian(nonwt_vals)) if len(nonwt_vals) else np.nan
        row["label_median__Non_WT_pooled"] = nonwt_med_pooled
        row["wt_vs_nonwt_abs_divergence"] = (
            abs(wt_med_pooled - nonwt_med_pooled)
            if (pd.notna(wt_med_pooled) and pd.notna(nonwt_med_pooled))
            else np.nan
        )

        if len(label_medians) == 0:
            row.update({
                "n_labels_with_data": 0, "n_labels_het_like": 0,
                "frac_labels_het_like": np.nan, "label_median_min": np.nan,
                "label_median_max": np.nan, "label_median_iqr": np.nan,
            })
        else:
            lm = np.asarray(label_medians, dtype=float)
            label_het = (lm >= 25) & (lm <= 75)
            row.update({
                "n_labels_with_data": int(len(lm)),
                "n_labels_het_like": int(label_het.sum()),
                "frac_labels_het_like": float(label_het.mean()),
                "label_median_min": float(np.nanmin(lm)),
                "label_median_max": float(np.nanmax(lm)),
                "label_median_iqr": float(np.nanpercentile(lm, 75) - np.nanpercentile(lm, 25)),
            })
        label_rows.append(row)

    feature_df = feature_df.merge(pd.DataFrame(label_rows), on="variant", how="left")

    label_median_cols = [c for c in feature_df.columns if c.startswith("label_median__")]
    base_feature_cols = [
        "valid_frac", "median_af", "mean_af", "af_iqr", "af_std", "frac_het_like",
        "frac_clean_het", "frac_ref_like", "frac_alt_like", "frac_fixed_like",
        "frac_labels_het_like", "label_median_min", "label_median_max", "label_median_iqr",
        "wt_vs_nonwt_abs_divergence",
    ]
    cluster_feature_cols = [c for c in base_feature_cols + label_median_cols if c in feature_df.columns]
    X = feature_df[cluster_feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        med = X[c].median(skipna=True)
        X[c] = X[c].fillna(med if np.isfinite(med) else 0)
    X_scaled = StandardScaler().fit_transform(X)

    n_variants = X_scaled.shape[0]
    max_k = min(P["max_k"], n_variants - 1)
    min_k = min(P["min_k"], max_k)
    silhouette_rows = []
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=P["n_init"])
        lab = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, lab) if len(set(lab)) > 1 else np.nan
        silhouette_rows.append({"k": k, "silhouette": sil})
    silhouette_df = pd.DataFrame(silhouette_rows)
    best_k = int(silhouette_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

    print("\nCluster K selection")
    print("-------------------")
    if show_tables:
        _display(silhouette_df)
    print(f"Selected k = {best_k}")

    km = KMeans(n_clusters=best_k, random_state=seed, n_init=P["n_init"])
    feature_df["af_cluster"] = km.fit_predict(X_scaled)

    variant_af_cluster_df = feature_df.copy()
    variant_af_cluster_df["_chr_sort"] = variant_af_cluster_df["variant"].map(lambda v: _chr_pos_key(v)[0])
    variant_af_cluster_df["_pos_sort"] = variant_af_cluster_df["variant"].map(lambda v: _chr_pos_key(v)[1])
    variant_af_cluster_df = (
        variant_af_cluster_df
        .sort_values(
            ["af_cluster", "is_filtered_germline_variant", "_chr_sort", "_pos_sort", "variant"],
            ascending=[True, False, True, True, True],
        )
        .reset_index(drop=True)
        .drop(columns=["_chr_sort", "_pos_sort"], errors="ignore")
    )
    variant_af_cluster_df.insert(0, "cluster_order_index", np.arange(1, variant_af_cluster_df.shape[0] + 1))

    af_cluster_summary_df = (
        variant_af_cluster_df.groupby("af_cluster")
        .agg(
            n_variants=("variant", "count"),
            n_filtered_germline_variants=("is_filtered_germline_variant", "sum"),
            n_other_or_candidate_variants=("is_other_or_candidate_variant", "sum"),
            median_af_cluster=("median_af", "median"),
            median_mean_af=("mean_af", "median"),
            median_af_iqr=("af_iqr", "median"),
            median_af_std=("af_std", "median"),
            median_frac_het_like=("frac_het_like", "median"),
            median_frac_clean_het=("frac_clean_het", "median"),
            median_frac_ref_like=("frac_ref_like", "median"),
            median_frac_alt_like=("frac_alt_like", "median"),
            median_frac_fixed_like=("frac_fixed_like", "median"),
            median_valid_frac=("valid_frac", "median"),
            median_n_valid=("n_valid", "median"),
            median_n_labels_with_data=("n_labels_with_data", "median"),
            median_frac_labels_het_like=("frac_labels_het_like", "median"),
        )
        .reset_index()
    )
    af_cluster_summary_df["filtered_germline_fraction_of_cluster"] = (
        af_cluster_summary_df["n_filtered_germline_variants"] / af_cluster_summary_df["n_variants"]
    )
    af_cluster_summary_df["filtered_germline_fraction_of_all_filtered"] = (
        af_cluster_summary_df["n_filtered_germline_variants"] / max(len(filtered_germline_vars_for_cluster), 1)
    )
    af_cluster_summary_df = af_cluster_summary_df.sort_values(
        ["n_filtered_germline_variants", "filtered_germline_fraction_of_cluster", "af_cluster"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    filtered_germline_by_cluster_df = (
        variant_af_cluster_df.loc[variant_af_cluster_df["is_filtered_germline_variant"]]
        .copy()
        .sort_values(["af_cluster", "variant"])
        .reset_index(drop=True)
    )
    filtered_germline_cluster_counts_df = (
        filtered_germline_by_cluster_df.groupby("af_cluster")
        .agg(
            n_filtered_germline_variants=("variant", "count"),
            filtered_germline_variants=("variant", lambda x: ";".join(sorted(pd.Series(x).astype(str).unique()))),
        )
        .reset_index()
        .sort_values(["n_filtered_germline_variants", "af_cluster"], ascending=[False, True])
        .reset_index(drop=True)
    )

    clustered_af_heatmap_variant_table = pd.DataFrame()
    af_cluster_color_key_df = pd.DataFrame()
    af_cluster_blocks_df = pd.DataFrame()

    def _plot_clustered_heatmap(variant_cluster_df):
        nonlocal clustered_af_heatmap_variant_table, af_cluster_color_key_df, af_cluster_blocks_df
        plot_df = variant_cluster_df.copy()
        plot_df["variant"] = plot_df["variant"].astype(str).str.strip()
        plot_df = plot_df.loc[plot_df["variant"].isin(af_mat.columns)].copy()
        if plot_df.empty:
            return
        plot_df = plot_df.sort_values(["cluster_order_index", "variant"]).reset_index(drop=True)
        plot_df.insert(0, "x_index", np.arange(1, plot_df.shape[0] + 1))
        plot_vars = plot_df["variant"].astype(str).tolist()
        af = af_mat[plot_vars].apply(pd.to_numeric, errors="coerce")
        cell_order = []
        for lab in label_order:
            idx = labels[labels.eq(lab)].index
            cell_order += af.loc[idx].mean(axis=1, skipna=True).sort_values().index.tolist()
        if not cell_order:
            cell_order = af.mean(axis=1, skipna=True).sort_values().index.tolist()
        af = af.loc[cell_order, plot_vars]
        n_cells, n_vars = af.shape
        cluster_ids = sorted(plot_df["af_cluster"].dropna().unique().tolist())
        cluster_to_color_index = {cl: i for i, cl in enumerate(cluster_ids)}
        plot_df["cluster_color_index"] = plot_df["af_cluster"].map(cluster_to_color_index).astype(int)
        cluster_bar = plot_df["cluster_color_index"].values.reshape(1, -1)
        n_clusters = len(cluster_ids)
        base_cmap = plt.get_cmap("tab20", n_clusters) if n_clusters <= 20 else plt.get_cmap("hsv", n_clusters)
        cluster_cmap = ListedColormap([base_cmap(i) for i in range(n_clusters)])
        af_cluster_color_key_df = pd.DataFrame({
            "af_cluster": cluster_ids,
            "cluster_color_index": [cluster_to_color_index[cl] for cl in cluster_ids],
            "cluster_color_hex": [to_hex(cluster_cmap(cluster_to_color_index[cl])) for cl in cluster_ids],
        })
        cluster_values = plot_df["af_cluster"].values
        blocks = []
        start = 0
        for i in range(1, len(cluster_values) + 1):
            if i == len(cluster_values) or cluster_values[i] != cluster_values[start]:
                cl = cluster_values[start]
                end = i - 1
                width = end - start + 1
                center = (start + end) / 2
                blocks.append({
                    "af_cluster": cl,
                    "start_x_index": start + 1,
                    "end_x_index": end + 1,
                    "start_zero_based": start,
                    "end_zero_based": end,
                    "width": width,
                    "center_zero_based": center,
                    "center_x_index": center + 1,
                })
                start = i
        af_cluster_blocks_df = pd.DataFrame(blocks)
        fig_width = max(9, min(38, 0.12 * n_vars + 7))
        fig_height = max(7.5, min(47, 0.045 * n_cells + 5.5))
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
        gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[0.65, 0.35, 10], width_ratios=[40, 0.7], hspace=0.03, wspace=0.04)
        ax_label = fig.add_subplot(gs[0, 0])
        ax_top = fig.add_subplot(gs[1, 0])
        ax = fig.add_subplot(gs[2, 0])
        cax = fig.add_subplot(gs[2, 1])
        fig.add_subplot(gs[0, 1]).axis("off")
        fig.add_subplot(gs[1, 1]).axis("off")
        ax_label.set_xlim(-0.5, n_vars - 0.5)
        ax_label.set_ylim(0, 1)
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        for spine in ax_label.spines.values():
            spine.set_visible(False)
        if P["show_cluster_names_on_top"]:
            for _, block in af_cluster_blocks_df.iterrows():
                if block["width"] >= P["min_cluster_label_width"]:
                    label_txt = f"{P['cluster_label_prefix']} {block['af_cluster']}" if block["width"] >= 8 else str(block["af_cluster"])
                    ax_label.text(block["center_zero_based"], 0.45, label_txt, ha="center", va="center", fontsize=8 if block["width"] >= 8 else 7, clip_on=False)
        title = f"{patient_id}: AF heatmap ordered by AF cluster" if patient_id is not None else "AF heatmap ordered by AF cluster"
        ax_label.set_title(title, fontsize=12, pad=4)
        ax_top.imshow(cluster_bar, aspect="auto", interpolation="nearest", cmap=cluster_cmap, vmin=-0.5, vmax=n_clusters - 0.5)
        ax_top.set_yticks([])
        ax_top.set_xticks([])
        ax_top.set_ylabel("AF\ncluster", rotation=0, ha="right", va="center", fontsize=8)
        for spine in ax_top.spines.values():
            spine.set_visible(False)
        im = ax.imshow(af.values, aspect="auto", interpolation="nearest", vmin=0, vmax=100, cmap="viridis")
        fig.colorbar(im, cax=cax).set_label("AF (%)")
        if n_vars <= P["max_xticks"]:
            tick_positions = np.arange(n_vars)
        else:
            tick_positions = np.unique(np.linspace(0, n_vars - 1, P["max_xticks"]).round().astype(int))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions + 1, fontsize=8)
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(f"Variant index, ordered by AF cluster (n={n_vars})")
        ax.set_ylabel(f"Cells ordered by label (n={n_cells})")
        if P["draw_cluster_boundaries"]:
            for i in range(1, len(cluster_values)):
                if cluster_values[i] != cluster_values[i - 1]:
                    b = i - 0.5
                    ax.axvline(b, linewidth=0.8, linestyle="-", alpha=0.6)
                    ax_top.axvline(b, linewidth=0.8, linestyle="-", alpha=0.6)
                    ax_label.axvline(b, linewidth=0.8, linestyle="-", alpha=0.6)
        plt.show()
        display_cols = [
            "x_index", "cluster_order_index", "variant", "af_cluster", "cluster_color_index",
            "is_filtered_germline_variant", "is_other_or_candidate_variant", "median_af", "mean_af",
            "af_iqr", "af_std", "frac_het_like", "frac_clean_het", "frac_ref_like", "frac_alt_like",
            "frac_fixed_like", "n_valid", "valid_frac", "n_labels_with_data", "frac_labels_het_like",
            "label_median_min", "label_median_max", "label_median_iqr",
            "label_median__Non_WT_pooled", "wt_vs_nonwt_abs_divergence",
        ]
        clustered_af_heatmap_variant_table = plot_df[[c for c in display_cols if c in plot_df.columns]].copy()
        if show_tables and P["show_cluster_color_key"]:
            print("\nAF cluster color key")
            _display(af_cluster_color_key_df)
        if show_tables:
            print("\nAF cluster block coordinates")
            _display(af_cluster_blocks_df)
            if P["show_heatmap_variant_table"]:
                print("\nClustered AF heatmap variant table")
                _display(clustered_af_heatmap_variant_table)

    if plot and P["plot_heatmap_before_summary"]:
        _plot_clustered_heatmap(variant_af_cluster_df)

    if show_tables:
        print("\nAF cluster summary")
        print("------------------")
        _display(af_cluster_summary_df)
        print("\nFiltered germline variants per AF cluster")
        print("-----------------------------------------")
        _display(filtered_germline_cluster_counts_df)
        print("\nFiltered germline variant assignments")
        print("-------------------------------------")
        _display(filtered_germline_by_cluster_df)
        print("\nAll variant AF-cluster assignments")
        print("----------------------------------")
        _display(variant_af_cluster_df)

    return {
        "variant_af_cluster_df": variant_af_cluster_df,
        "af_cluster_summary_df": af_cluster_summary_df,
        "filtered_germline_by_cluster_df": filtered_germline_by_cluster_df,
        "filtered_germline_cluster_counts_df": filtered_germline_cluster_counts_df,
        "silhouette_df": silhouette_df,
        "clustered_af_heatmap_variant_table": clustered_af_heatmap_variant_table,
        "af_cluster_color_key_df": af_cluster_color_key_df,
        "af_cluster_blocks_df": af_cluster_blocks_df,
        "af_mat": af_mat,
        "labels": labels,
        "label_order": label_order,
        "cluster_feature_cols": cluster_feature_cols,
        "params": P,
    }


def rename_af_clusters_and_update_filtered_germline(
    *,
    variant_af_cluster_df,
    af_mat,
    df_AF_all,
    filtered_germline_variants,
    af_cluster_rename_map=None,
    all_variant_germline_df=None,
    df=None,
    df_review=None,
    labels=None,
    patient_id: str = None,
    # Optional dataframe from CNV_Utils.extract_cnv_data():
    # out_all["df_mut_to_amplicon_dataframe"] with columns variant, amplicon.
    # If omitted, the function resolves this automatically from the active
    # notebook/script namespace (including out_all) before using positional
    # fallback blocks. An explicit argument still has highest priority.
    df_mut_to_amplicon_dataframe=None,
    germline_renamed_cluster_name: str = "germline_het",
    excluded_renamed_cluster_name: str = "excluded_blacklisted_pathogenic",
    discarded_poor_qc_renamed_cluster_name: str = "discarded_poor_qc",
    loh_cluster_ids=None,
    exclude_blacklisted: bool = True,
    exclude_pathogenic: bool = True,
    force_original_filtered_germline: bool = True,
    preserve_baseline_germline_evidence: bool = True,
    require_baseline_germline_evidence: bool = True,
    baseline_germline_evidence_cols=None,
    baseline_het_genotype_values=("het", "heterozygous"),
    allow_input_filtered_germline_as_baseline_evidence: bool = False,
    allow_input_filtered_germline_as_loh_evidence: bool = True,
    allow_non_snv_germline_markers: bool = False,
    exclude_systematically_skewed_baseline_markers: bool = True,
    systematic_skew_low: float = 30.0,
    systematic_skew_high: float = 70.0,
    systematic_skew_max_label_median_iqr: float = 10.0,
    systematic_skew_max_label_median_range: float = 15.0,
    systematic_skew_min_labels_with_data: int = 2,
    systematic_skew_min_frac_labels_het_like: float = 2 / 3,
    protect_preliminary_loh_variants_from_systematic_skew_filter: bool = True,

    # Local linked-artifact block gate. This is deliberately separate from the
    # single-locus systematic-skew filter: it identifies clusters of nearby,
    # rescued-only SNVs with near-identical AF traces that are unsuitable as
    # independent germline/LOH markers.
    exclude_local_linked_artifact_blocks: bool = True,
    local_artifact_min_block_variants: int = 2,
    local_artifact_max_block_span_bp: int = 25,
    local_artifact_require_same_raw_af_cluster: bool = True,
    local_artifact_low: float = 35.0,
    local_artifact_high: float = 65.0,
    # A same-amplicon block is only an artifact when every member is
    # consistently skewed across clone labels, rather than showing the
    # label-to-label divergence expected for a clone-specific LOH event.
    # Per-cell AF-profile concordance is optional: the group-level label
    # behaviour is the primary evidence for an amplicon-level artifact.
    local_artifact_require_low_label_median_range: bool = True,
    local_artifact_max_label_median_range: float = 8.0,
    local_artifact_require_same_amplicon: bool = True,
    # Same-amplicon evidence is required by default. Positional proximity is
    # never treated as a substitute for amplicon membership unless explicitly
    # enabled for an audit-only legacy run.
    local_artifact_allow_position_fallback_without_amplicon_mapping: bool = False,
    local_artifact_min_pairwise_shared_cells: int = 50,
    local_artifact_min_pairwise_spearman: float = 0.98,
    local_artifact_max_pairwise_median_abs_diff: float = 3.0,
    # Pairwise AF concordance remains a diagnostic. It is intentionally NOT
    # required by default: a same-amplicon, rescued-only block that is
    # uniformly AF-skewed across labels and not LOH is already unsuitable as
    # an independent germline marker, even if per-cell AF values differ due to
    # allele dropout or depth noise.
    local_artifact_require_profile_concordance: bool = False,
    local_artifact_require_rescued_not_clean: bool = True,
    # Same-amplicon blocks are the only default artifact grouping. A
    # position-based fallback remains available for legacy inspection but is
    # disabled by default because genomic adjacency does not establish shared
    # technical origin.
    local_artifact_use_amplicon_blocks: bool = True,
    local_artifact_fallback_to_position_blocks: bool = False,

    plot: bool = True,
    include_variant_tables: bool = True,
    max_xticks: int = 60,
    verbose: bool = True,
):
    """Rename AF clusters and update the usable germline-marker list.

    This function deliberately separates two concepts:

    * ``germline_het`` is a *validated germline marker* class.  A raw AF
      cluster mapped to that class is insufficient by itself: a variant must
      also have baseline germline-HET evidence.
    * ``is_loh_variant`` is a parallel LOH tracking flag.  A validated marker
      may be strongly LOH-distorted in the current sample (median AF close to
      0 or 100) and remains a germline marker when its baseline evidence is
      present.

    Consequently, there is no global low-/high-median AF exclusion.  Instead,
    cluster-only HET assignments that lack baseline evidence are reassigned to
    ``discarded_poor_qc_renamed_cluster_name``.  LOH tracking uses a distinct
    evidence gate: a simple-SNV marker in a designated raw LOH cluster may use
    its original filtered-germline provenance as baseline support, while
    non-LOH ``germline_het`` assignments still require independent evidence.

    Baseline evidence must be independent of the current AF-cluster call.
    By default, membership in the input ``filtered_germline_variants`` is
    treated as provenance only, not proof: an upstream list can contain an
    old false-positive marker and must not preserve itself circularly.

    Independent evidence is any of:
      1. a calibrated / clean / rescued / concordant germline seed;
      2. a ``donor_genotype`` value indicating HET; or
      3. a validated/rescued germline broad class.

    ``allow_input_filtered_germline_as_baseline_evidence=False`` is the
    conservative default. Set it to True only for a legacy run where no
    all-variant evidence table is available.

    Only simple single-nucleotide alleles (A/C/G/T -> A/C/G/T) are accepted
    as germline/LOH markers by default. Indels, symbolic alleles, and malformed
    representations are assigned ``discarded_poor_qc``.

    A separate systematic-bias gate prevents an upstream baseline seed from
    preserving a locus that is consistently near-reference or near-alt across
    all observed labels. Such loci are not clone-specific LOH markers: they
    have little between-label AF divergence and too little HET-like label
    support. Raw clusters explicitly tagged as preliminary LOH are protected by
    default so genuine LOH markers are not discarded solely because an event is
    broad in the current sample.

    Parameters
    ----------
    preserve_baseline_germline_evidence : bool, default True
        Preserve baseline-supported variants as ``germline_het`` regardless of
        their current raw AF cluster. This protects genuine markers distorted
        by a large LOH clone. Hard exclusions always take precedence.
    require_baseline_germline_evidence : bool, default True
        Send raw-cluster-only ``germline_het`` assignments with no baseline
        evidence to ``discarded_poor_qc_renamed_cluster_name``.
    baseline_germline_evidence_cols : iterable[str], optional
        Additional boolean columns whose true values should count as baseline
        germline evidence. Canonical seed columns are always included.
    loh_cluster_ids : iterable, optional
        Raw AF cluster ids to tag as LOH. A marker can be both germline and LOH
        only when it passes the baseline-evidence gate. Poor-QC assignments are
        never returned as LOH markers.
    allow_input_filtered_germline_as_baseline_evidence : bool, default False
        Allow legacy membership in ``filtered_germline_variants`` to preserve a
        non-LOH marker even without independent genotype/seed evidence. Disabled
        by default to prevent circular retention of old false positives.
    allow_input_filtered_germline_as_loh_evidence : bool, default True
        For raw clusters explicitly listed in ``loh_cluster_ids``, accept
        original filtered-germline provenance as sufficient baseline evidence
        for a simple-SNV LOH marker. This does not relax the baseline gate for
        non-LOH markers, and it never admits unsupported variants, non-SNVs, or
        hard-excluded markers.
    allow_non_snv_germline_markers : bool, default False
        Keep non-SNV alleles as germline markers. Disabled by default because
        AF is not reliably diploid-interpretable for indels/symbolic alleles.
    exclude_systematically_skewed_baseline_markers : bool, default True
        Discard a baseline-supported simple-SNV marker when every observed
        label remains consistently outside the broad HET AF interval. This
        catches systematic reference/alternate bias that upstream seed columns
        alone cannot distinguish from a donor HET.
    systematic_skew_low, systematic_skew_high : float
        Lower/upper bounds of the broad label-median HET interval. A marker is
        systematically skewed only when all observed label medians are below
        ``systematic_skew_low`` or all are above
        ``systematic_skew_high``.
    systematic_skew_max_label_median_iqr, systematic_skew_max_label_median_range : float
        Maximum allowed between-label IQR and full range for a systematic-bias
        call. Genuine clone-specific LOH should show greater label divergence.
    systematic_skew_min_labels_with_data : int
        Minimum number of labels with valid AF data required to make this
        cross-label decision.
    systematic_skew_min_frac_labels_het_like : float
        Markers with at least this fraction of labels in the 25--75% HET-like
        band are retained; lower fractions are eligible for systematic-bias
        exclusion.
    protect_preliminary_loh_variants_from_systematic_skew_filter : bool, default True
        Preserve raw clusters named in ``loh_cluster_ids`` from the systematic
        bias gate. They remain subject to all other germline/LOH QC.
    local_artifact_require_low_label_median_range : bool, default True
        Require every member of a same-amplicon artifact block to have a small
        clone-label median AF range; this identifies technical AF skew rather
        than clone-specific LOH.
    local_artifact_max_label_median_range : float, default 8.0
        Largest permitted per-variant ``label_median_range`` in a bad block.
    local_artifact_require_same_amplicon : bool, default True
        Require a verified shared amplicon for poor-QC assignment when a
        usable variant-to-amplicon mapping is available.
    local_artifact_allow_position_fallback_without_amplicon_mapping : bool, default False
        Legacy diagnostic option. When True together with
        ``local_artifact_fallback_to_position_blocks=True``, a positional
        block may be evaluated if no mapping is available. It is disabled by
        default because genomic proximity is not evidence of shared amplicon
        origin.
    df_mut_to_amplicon_dataframe : pd.DataFrame, optional
        Variant-to-amplicon mapping returned by
        ``CNV_Utils.extract_cnv_data(...)["df_mut_to_amplicon_dataframe"]``.
        It must contain ``variant`` and ``amplicon`` columns. When omitted,
        this function automatically looks in the active notebook/script
        namespace for ``df_mut_to_amplicon_all_dataframe``,
        ``df_mut_to_amplicon_dataframe``, or
        ``out_all["df_mut_to_amplicon_dataframe"]``. It also reconstructs the
        map from a live ``Sample`` / ``Sample_all_vars`` object using
        ``Sample.dna.col_attrs['id']`` and ``['amplicon']``; when that direct
        annotation is absent, it uses CNV amplicon intervals from
        ``Sample.cnv.col_attrs``. Local artifact blocks are defined by shared
        assay amplicon, not by genomic distance.
    exclude_local_linked_artifact_blocks : bool, default True
        Reassign a questionable multi-SNV block to ``discarded_poor_qc`` only
        when its members share the same verified assay amplicon, share the
        same raw AF cluster, are rescued-but-not-clean baseline markers, have
        uniformly low or high label medians with low within-variant
        label-median ranges, and have no LOH designation. This is an
        amplicon-quality gate; positional proximity is not used by default.
    local_artifact_min_block_variants, local_artifact_max_block_span_bp
        Minimum block size and maximum total genomic span for the optional
        legacy positional
        fallback. Same-amplicon groups do not require a small genomic span: the
        shared assay target supplies the local linkage definition.
    local_artifact_low, local_artifact_high
        Low/high label-median bounds used only by this block-level gate.
    local_artifact_min_pairwise_shared_cells,
    local_artifact_min_pairwise_spearman,
    local_artifact_max_pairwise_median_abs_diff
        Pairwise AF-profile concordance requirements across all block members.
        Every pair must have sufficient jointly observed cells, high Spearman
        correlation, and little median absolute AF difference.
    local_artifact_require_rescued_not_clean : bool, default True
        Restrict the block gate to rescued markers that lack a clean-marker
        seed. This prevents the local rule from discarding adjacent,
        high-confidence clean germline markers.
    local_artifact_use_amplicon_blocks : bool, default True
        Use same-amplicon candidate groups whenever a variant-to-amplicon map
        is supplied.
    local_artifact_fallback_to_position_blocks : bool, default True
        Evaluate positional windows only for candidate variants that lack an
        amplicon mapping. This retains backwards-compatible behavior when the
        CNV mapping is unavailable without double-evaluating mapped variants.
    """
    import re
    import matplotlib.pyplot as plt

    if not (0 <= float(systematic_skew_low) < float(systematic_skew_high) <= 100):
        raise ValueError(
            "systematic_skew_low/high must satisfy 0 <= low < high <= 100."
        )
    if float(systematic_skew_max_label_median_iqr) < 0:
        raise ValueError(
            "systematic_skew_max_label_median_iqr must be non-negative."
        )
    if float(systematic_skew_max_label_median_range) < 0:
        raise ValueError(
            "systematic_skew_max_label_median_range must be non-negative."
        )
    if float(local_artifact_max_label_median_range) < 0:
        raise ValueError(
            "local_artifact_max_label_median_range must be non-negative."
        )
    if int(systematic_skew_min_labels_with_data) < 1:
        raise ValueError(
            "systematic_skew_min_labels_with_data must be at least 1."
        )
    if not (0 <= float(systematic_skew_min_frac_labels_het_like) <= 1):
        raise ValueError(
            "systematic_skew_min_frac_labels_het_like must be between 0 and 1."
        )
    if int(local_artifact_min_block_variants) < 2:
        raise ValueError(
            "local_artifact_min_block_variants must be at least 2."
        )
    if int(local_artifact_max_block_span_bp) < 0:
        raise ValueError(
            "local_artifact_max_block_span_bp must be non-negative."
        )
    if not (0 <= float(local_artifact_low) < float(local_artifact_high) <= 100):
        raise ValueError(
            "local_artifact_low/high must satisfy 0 <= low < high <= 100."
        )
    if int(local_artifact_min_pairwise_shared_cells) < 1:
        raise ValueError(
            "local_artifact_min_pairwise_shared_cells must be at least 1."
        )
    if not (-1 <= float(local_artifact_min_pairwise_spearman) <= 1):
        raise ValueError(
            "local_artifact_min_pairwise_spearman must be between -1 and 1."
        )
    if float(local_artifact_max_pairwise_median_abs_diff) < 0:
        raise ValueError(
            "local_artifact_max_pairwise_median_abs_diff must be non-negative."
        )

    rename_map = dict(
        DEFAULT_AF_CLUSTER_RENAME_MAP
        if af_cluster_rename_map is None
        else af_cluster_rename_map
    )

    def _normalise_af_cluster_id(x):
        if pd.isna(x):
            return x
        try:
            xf = float(x)
            if xf.is_integer():
                return int(xf)
        except Exception:
            pass
        return str(x).strip()

    rename_map = {
        _normalise_af_cluster_id(k): str(v)
        for k, v in rename_map.items()
    }

    if (
        "variant" not in variant_af_cluster_df.columns
        or "af_cluster" not in variant_af_cluster_df.columns
    ):
        raise KeyError(
            "variant_af_cluster_df must contain 'variant' and 'af_cluster'."
        )
    if not isinstance(af_mat, pd.DataFrame):
        raise TypeError("af_mat must be a pandas DataFrame.")

    if labels is None:
        labels = (
            df_AF_all["label"].astype(str)
            if "label" in df_AF_all.columns
            else pd.Series("all_cells", index=df_AF_all.index)
        )
    else:
        labels = pd.Series(labels, index=getattr(labels, "index", None)).astype(str)

    original_filtered = _extract_variant_list(
        filtered_germline_variants,
        "filtered_germline_variants",
    )
    original_filtered_set = {
        str(v) for v in original_filtered if str(v) in af_mat.columns
    }

    renamed_cluster_variant_df = variant_af_cluster_df.copy()
    renamed_cluster_variant_df["variant"] = (
        renamed_cluster_variant_df["variant"].astype(str).str.strip()
    )
    renamed_cluster_variant_df = renamed_cluster_variant_df.loc[
        renamed_cluster_variant_df["variant"].isin(af_mat.columns)
    ].copy()
    renamed_cluster_variant_df["raw_af_cluster"] = (
        renamed_cluster_variant_df["af_cluster"]
    )
    renamed_cluster_variant_df["raw_af_cluster_norm"] = (
        renamed_cluster_variant_df["raw_af_cluster"].map(_normalise_af_cluster_id)
    )
    renamed_cluster_variant_df["renamed_cluster_from_dict"] = (
        renamed_cluster_variant_df["raw_af_cluster_norm"].map(rename_map)
    )
    renamed_cluster_variant_df["renamed_cluster"] = (
        renamed_cluster_variant_df["renamed_cluster_from_dict"]
        .fillna(
            renamed_cluster_variant_df["raw_af_cluster_norm"].map(
                lambda x: f"cluster_{x}"
            )
        )
        .astype(str)
    )
    renamed_cluster_variant_df["was_original_filtered_germline"] = (
        renamed_cluster_variant_df["variant"].isin(original_filtered_set)
    )

    def _simple_snv_qc(variant):
        """Return (is_simple_snv, reason) for a Tapestri variant ID.

        Only biallelic A/C/G/T single-nucleotide substitutions are valid for
        AF-based germline/LOH marker use. This excludes indels, ``*`` alleles,
        ``+`` representations, multi-base alleles, and malformed IDs without
        relying on per-variant hard-coded lists.
        """
        m = re.match(r"^chr[^:]+:\d+:([^/]+)/([^/]+)$", str(variant).strip())
        if m is None:
            return False, "malformed_variant_id"
        ref, alt = m.group(1).upper(), m.group(2).upper()
        if len(ref) != 1 or len(alt) != 1:
            return False, "non_snv_indel_or_complex_allele"
        if ref not in {"A", "C", "G", "T"} or alt not in {"A", "C", "G", "T"}:
            return False, "symbolic_or_non_acgt_allele"
        if ref == alt:
            return False, "non_variant_same_ref_alt"
        return True, "simple_snv"

    _allele_qc = renamed_cluster_variant_df["variant"].map(_simple_snv_qc)
    renamed_cluster_variant_df["is_simple_snv"] = _allele_qc.map(lambda x: bool(x[0]))
    renamed_cluster_variant_df["variant_allele_qc_reason"] = _allele_qc.map(lambda x: str(x[1]))

    def _source_variant_col(src):
        for c in [
            "variant", "Variant", "id", "ID", "variant_id", "Variant_ID",
        ]:
            if c in src.columns:
                return c
        return None

    def _contains_blacklist(series):
        s = series.astype(str).str.lower()
        neg = s.str.contains(
            r"\b(?:no|not|non)[_\-\s]*blacklist(?:ed)?\b",
            regex=True,
            na=False,
        )
        pos = s.str.contains(r"blacklist|blacklisted", regex=True, na=False)
        return pos & ~neg

    def _contains_pathogenic(series):
        s = series.astype(str).str.lower()
        neg = s.str.contains(
            r"\b(?:no|not|non)[_\-\s]*pathogenic\b",
            regex=True,
            na=False,
        )
        pos = s.str.contains(
            r"\bpathogenic\b|likely[_\-\s]*pathogenic",
            regex=True,
            na=False,
        )
        return pos & ~neg

    def _flags_from_variant_dataframe(src, source_name):
        cols = [
            "variant",
            f"hard_excluded_blacklisted__{source_name}",
            f"hard_excluded_pathogenic__{source_name}",
        ]
        if src is None or not isinstance(src, pd.DataFrame) or src.empty:
            return pd.DataFrame(columns=cols)
        var_col = _source_variant_col(src)
        if var_col is None:
            return pd.DataFrame(columns=cols)

        x = src.copy()
        x["variant"] = _variant_key_series(x[var_col])
        black = pd.Series(False, index=x.index)
        path = pd.Series(False, index=x.index)
        annotation_name_tokens = [
            "status", "verdict", "reason", "category", "class",
            "classification", "filter", "flag", "annotation", "comment",
            "clinvar", "clinical", "miss", "germline_broad_class",
        ]
        for c in x.columns:
            if c == "variant":
                continue
            cname = str(c).lower()
            if any(
                neg in cname
                for neg in [
                    "not_black", "non_black", "not_pathogenic",
                    "non_pathogenic",
                ]
            ):
                continue
            is_black_col = "blacklist" in cname or "blacklisted" in cname
            is_path_col = "pathogenic" in cname
            is_annotation_col = any(
                tok in cname for tok in annotation_name_tokens
            )
            if is_black_col:
                black = black | _coerce_bool_like(x[c]) | _contains_blacklist(x[c])
            if is_path_col:
                path = path | _coerce_bool_like(x[c]) | _contains_pathogenic(x[c])
            if is_annotation_col:
                black = black | _contains_blacklist(x[c])
                path = path | _contains_pathogenic(x[c])

        out = pd.DataFrame({
            "variant": x["variant"],
            f"hard_excluded_blacklisted__{source_name}": (
                black.fillna(False).astype(bool)
            ),
            f"hard_excluded_pathogenic__{source_name}": (
                path.fillna(False).astype(bool)
            ),
        })
        return out.groupby("variant", as_index=False).agg("max")

    # A true germline marker must be supported independently of both the
    # current AF cluster and the legacy input list. The first group below is
    # strong/independent evidence. The second is retained for audit only: these
    # flags can be derived from an earlier marker list and must not preserve a
    # variant by themselves.
    canonical_independent_bool_cols = [
        "calibrated_usable_het_seed",
        "calibrated_clean_het_seed",
        "calibrated_rescued_het_seed",
        "clean_germline_het_seed",
        "rescued_germline_het_seed",
        "concordant_germline_het",
    ]
    canonical_soft_bool_cols = [
        "germline_het_seed",
        "usable_germline_het_seed",
    ]
    additional_independent_cols = list(baseline_germline_evidence_cols or [])
    independent_bool_cols = list(
        dict.fromkeys(canonical_independent_bool_cols + additional_independent_cols)
    )
    baseline_bool_cols = list(
        dict.fromkeys(independent_bool_cols + canonical_soft_bool_cols)
    )
    baseline_het_genotype_values_norm = {
        str(x).strip().lower() for x in baseline_het_genotype_values
    }
    validated_broad_classes = {
        "clean_germline_het",
        "rescued_germline_het_candidate",
        "rescued_germline_het_whitelist_low_evidence",
    }

    def _baseline_evidence_from_variant_dataframe(src, source_name):
        cols = [
            "variant",
            f"baseline_germline_evidence__{source_name}",
            f"baseline_germline_evidence_sources__{source_name}",
            f"independent_baseline_germline_evidence__{source_name}",
            f"independent_baseline_germline_evidence_sources__{source_name}",
        ]
        if src is None or not isinstance(src, pd.DataFrame) or src.empty:
            return pd.DataFrame(columns=cols)
        var_col = _source_variant_col(src)
        if var_col is None:
            return pd.DataFrame(columns=cols)

        x = src.copy()
        x["variant"] = _variant_key_series(x[var_col])
        evidence = pd.Series(False, index=x.index)
        evidence_source_text = pd.Series("", index=x.index, dtype=object)
        independent = pd.Series(False, index=x.index)
        independent_source_text = pd.Series("", index=x.index, dtype=object)

        def _append_source(current, hit, token):
            arr = np.where(
                hit,
                np.where(current.eq(""), token, current + ";" + token),
                current,
            )
            return pd.Series(arr, index=x.index, dtype=object)

        for c in baseline_bool_cols:
            if c not in x.columns:
                continue
            hit = _coerce_bool_like(x[c]).fillna(False).astype(bool)
            evidence = evidence | hit
            evidence_source_text = _append_source(
                evidence_source_text, hit, f"{source_name}:{c}"
            )
            if c in independent_bool_cols:
                independent = independent | hit
                independent_source_text = _append_source(
                    independent_source_text, hit, f"{source_name}:{c}"
                )

        if "donor_genotype" in x.columns:
            donor_het = (
                x["donor_genotype"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(baseline_het_genotype_values_norm)
            )
            evidence = evidence | donor_het
            independent = independent | donor_het
            evidence_source_text = _append_source(
                evidence_source_text, donor_het, f"{source_name}:donor_genotype=het"
            )
            independent_source_text = _append_source(
                independent_source_text, donor_het, f"{source_name}:donor_genotype=het"
            )

        if "germline_broad_class" in x.columns:
            broad_het = (
                x["germline_broad_class"]
                .astype(str)
                .str.strip()
                .isin(validated_broad_classes)
            )
            evidence = evidence | broad_het
            independent = independent | broad_het
            evidence_source_text = _append_source(
                evidence_source_text, broad_het, f"{source_name}:germline_broad_class"
            )
            independent_source_text = _append_source(
                independent_source_text, broad_het, f"{source_name}:germline_broad_class"
            )

        out = pd.DataFrame({
            "variant": x["variant"],
            f"baseline_germline_evidence__{source_name}": evidence.astype(bool),
            f"baseline_germline_evidence_sources__{source_name}": evidence_source_text,
            f"independent_baseline_germline_evidence__{source_name}": independent.astype(bool),
            f"independent_baseline_germline_evidence_sources__{source_name}": independent_source_text,
        })

        def _join_unique(values):
            vals = [str(v) for v in values if pd.notna(v) and str(v).strip()]
            return ";".join(dict.fromkeys(vals))

        return (
            out.groupby("variant", as_index=False)
            .agg({
                f"baseline_germline_evidence__{source_name}": "max",
                f"baseline_germline_evidence_sources__{source_name}": _join_unique,
                f"independent_baseline_germline_evidence__{source_name}": "max",
                f"independent_baseline_germline_evidence_sources__{source_name}": _join_unique,
            })
        )

    annotation_sources = [("variant_af_cluster_df", variant_af_cluster_df)]
    if df is not None and isinstance(df, pd.DataFrame):
        annotation_sources.append(("df", df))
    if (
        all_variant_germline_df is not None
        and isinstance(all_variant_germline_df, pd.DataFrame)
    ):
        annotation_sources.append(
            ("all_variant_germline_df", all_variant_germline_df)
        )
    if df_review is not None and isinstance(df_review, pd.DataFrame):
        annotation_sources.append(("df_review", df_review))

    for source_name, source_df in annotation_sources:
        flag_df = _flags_from_variant_dataframe(source_df, source_name)
        if not flag_df.empty:
            renamed_cluster_variant_df = renamed_cluster_variant_df.merge(
                flag_df,
                on="variant",
                how="left",
            )

        evidence_df = _baseline_evidence_from_variant_dataframe(
            source_df,
            source_name,
        )
        if not evidence_df.empty:
            renamed_cluster_variant_df = renamed_cluster_variant_df.merge(
                evidence_df,
                on="variant",
                how="left",
            )

    black_flag_cols = [
        c for c in renamed_cluster_variant_df.columns
        if c.startswith("hard_excluded_blacklisted__")
    ]
    path_flag_cols = [
        c for c in renamed_cluster_variant_df.columns
        if c.startswith("hard_excluded_pathogenic__")
    ]
    for c in black_flag_cols + path_flag_cols:
        renamed_cluster_variant_df[c] = (
            renamed_cluster_variant_df[c].fillna(False).astype(bool)
        )

    renamed_cluster_variant_df["hard_excluded_blacklisted"] = (
        renamed_cluster_variant_df[black_flag_cols].any(axis=1)
        if exclude_blacklisted and black_flag_cols
        else False
    )
    renamed_cluster_variant_df["hard_excluded_pathogenic"] = (
        renamed_cluster_variant_df[path_flag_cols].any(axis=1)
        if exclude_pathogenic and path_flag_cols
        else False
    )
    renamed_cluster_variant_df[
        "hard_excluded_blacklisted_or_pathogenic"
    ] = (
        renamed_cluster_variant_df["hard_excluded_blacklisted"]
        | renamed_cluster_variant_df["hard_excluded_pathogenic"]
    )

    baseline_evidence_cols_merged = [
        c for c in renamed_cluster_variant_df.columns
        if c.startswith("baseline_germline_evidence__")
    ]
    baseline_source_cols_merged = [
        c for c in renamed_cluster_variant_df.columns
        if c.startswith("baseline_germline_evidence_sources__")
    ]
    independent_evidence_cols_merged = [
        c for c in renamed_cluster_variant_df.columns
        if c.startswith("independent_baseline_germline_evidence__")
    ]
    independent_source_cols_merged = [
        c for c in renamed_cluster_variant_df.columns
        if c.startswith("independent_baseline_germline_evidence_sources__")
    ]
    for c in baseline_evidence_cols_merged + independent_evidence_cols_merged:
        renamed_cluster_variant_df[c] = (
            renamed_cluster_variant_df[c].fillna(False).astype(bool)
        )
    for c in baseline_source_cols_merged + independent_source_cols_merged:
        renamed_cluster_variant_df[c] = (
            renamed_cluster_variant_df[c].fillna("").astype(str)
        )

    renamed_cluster_variant_df["has_input_filtered_germline_provenance"] = (
        renamed_cluster_variant_df["was_original_filtered_germline"].fillna(False)
    )
    renamed_cluster_variant_df["has_any_upstream_germline_support"] = (
        renamed_cluster_variant_df["has_input_filtered_germline_provenance"]
        | (
            renamed_cluster_variant_df[baseline_evidence_cols_merged].any(axis=1)
            if baseline_evidence_cols_merged
            else False
        )
    )
    renamed_cluster_variant_df["has_independent_baseline_germline_evidence"] = (
        renamed_cluster_variant_df[independent_evidence_cols_merged].any(axis=1)
        if independent_evidence_cols_merged
        else False
    )
    renamed_cluster_variant_df["has_baseline_germline_evidence"] = (
        renamed_cluster_variant_df["has_independent_baseline_germline_evidence"]
        | (
            bool(allow_input_filtered_germline_as_baseline_evidence)
            & renamed_cluster_variant_df["has_input_filtered_germline_provenance"]
        )
    )

    # This field is deliberately broader than
    # ``has_baseline_germline_evidence`` only for designated LOH clusters.
    # It preserves an original, upstream-filtered simple-SNV marker when LOH
    # shifts its current AF far from the germline-HET regime, without allowing
    # old input-list membership to rescue ordinary non-LOH cluster assignments.
    renamed_cluster_variant_df["has_loh_baseline_germline_evidence"] = (
        renamed_cluster_variant_df["has_independent_baseline_germline_evidence"]
        | (
            bool(allow_input_filtered_germline_as_loh_evidence)
            & renamed_cluster_variant_df["has_input_filtered_germline_provenance"]
        )
    )

    def _collect_sources(row, source_cols, include_input=False):
        sources = []
        if include_input and bool(row.get("has_input_filtered_germline_provenance", False)):
            sources.append("input_filtered_germline_variants")
        for c in source_cols:
            value = str(row.get(c, "") or "").strip()
            if value:
                sources.extend([x for x in value.split(";") if x])
        return ";".join(dict.fromkeys(sources))

    renamed_cluster_variant_df["baseline_germline_evidence_sources"] = (
        renamed_cluster_variant_df.apply(
            lambda r: _collect_sources(
                r, baseline_source_cols_merged, include_input=True
            ),
            axis=1,
        )
    )
    renamed_cluster_variant_df[
        "independent_baseline_germline_evidence_sources"
    ] = (
        renamed_cluster_variant_df.apply(
            lambda r: _collect_sources(
                r, independent_source_cols_merged, include_input=False
            ),
            axis=1,
        )
    )

    # Parallel LOH eligibility is calculated before poor-QC reassignment.
    # ``was_preliminary_loh_variant`` records the raw-cluster designation for
    # audit. ``is_loh_variant`` is stricter: it requires a simple SNV, no hard
    # exclusion, and either independent baseline evidence or original
    # filtered-germline provenance (the latter only under the explicit LOH
    # evidence policy above).
    if loh_cluster_ids is None:
        loh_cluster_ids_norm = set()
    else:
        loh_cluster_ids_norm = {
            _normalise_af_cluster_id(c) for c in loh_cluster_ids
        }

    renamed_cluster_variant_df["was_preliminary_loh_variant"] = (
        renamed_cluster_variant_df["raw_af_cluster_norm"].isin(
            loh_cluster_ids_norm
        )
        & ~renamed_cluster_variant_df[
            "hard_excluded_blacklisted_or_pathogenic"
        ]
    )
    renamed_cluster_variant_df["is_loh_variant"] = (
        renamed_cluster_variant_df["was_preliminary_loh_variant"]
        & renamed_cluster_variant_df[
            "has_loh_baseline_germline_evidence"
        ].fillna(False)
        & renamed_cluster_variant_df["is_simple_snv"].fillna(False)
        & ~renamed_cluster_variant_df[
            "hard_excluded_blacklisted_or_pathogenic"
        ].fillna(False)
    )

    def _hard_exclusion_reason(row):
        reasons = []
        if bool(row.get("hard_excluded_blacklisted", False)):
            reasons.append("blacklisted")
        if bool(row.get("hard_excluded_pathogenic", False)):
            reasons.append("pathogenic")
        return ";".join(reasons)

    def _hard_exclusion_sources(row):
        sources = []
        for c in black_flag_cols:
            if bool(row.get(c, False)):
                sources.append(
                    c.replace("hard_excluded_blacklisted__", "blacklisted:")
                )
        for c in path_flag_cols:
            if bool(row.get(c, False)):
                sources.append(
                    c.replace("hard_excluded_pathogenic__", "pathogenic:")
                )
        return ";".join(sources)

    renamed_cluster_variant_df["hard_exclusion_reason"] = (
        renamed_cluster_variant_df.apply(_hard_exclusion_reason, axis=1)
    )
    renamed_cluster_variant_df["hard_exclusion_sources"] = (
        renamed_cluster_variant_df.apply(_hard_exclusion_sources, axis=1)
    )

    # Retain independently supported simple-SNV markers even when their
    # current AF shape has shifted into an LOH-like raw cluster. In addition,
    # validated LOH markers can use original filtered-germline provenance under
    # the LOH-specific policy; this exception never applies to non-LOH markers.
    baseline_preserve_mask = (
        renamed_cluster_variant_df["has_baseline_germline_evidence"]
        & (
            renamed_cluster_variant_df["is_simple_snv"]
            | bool(allow_non_snv_germline_markers)
        )
        & ~renamed_cluster_variant_df[
            "hard_excluded_blacklisted_or_pathogenic"
        ]
    )
    loh_preserve_mask = renamed_cluster_variant_df["is_loh_variant"].fillna(False)
    if preserve_baseline_germline_evidence:
        renamed_cluster_variant_df.loc[
            baseline_preserve_mask | loh_preserve_mask,
            "renamed_cluster",
        ] = germline_renamed_cluster_name
    elif force_original_filtered_germline:
        force_mask = (
            renamed_cluster_variant_df["was_original_filtered_germline"]
            & (
                renamed_cluster_variant_df["is_simple_snv"]
                | bool(allow_non_snv_germline_markers)
            )
            & ~renamed_cluster_variant_df[
                "hard_excluded_blacklisted_or_pathogenic"
            ]
        )
        renamed_cluster_variant_df.loc[
            force_mask,
            "renamed_cluster",
        ] = germline_renamed_cluster_name

    # A raw AF cluster mapping, a provisional LOH cluster, or old list
    # membership is never enough to establish a usable marker. The sole
    # exception is a validated LOH marker: it must be a simple SNV in a
    # designated LOH cluster with either independent evidence or original
    # filtered-germline provenance.
    provisional_marker_candidate_mask = (
        (
            renamed_cluster_variant_df["renamed_cluster"].eq(
                germline_renamed_cluster_name
            )
            | renamed_cluster_variant_df["was_original_filtered_germline"]
            | renamed_cluster_variant_df["was_preliminary_loh_variant"]
        )
        & ~renamed_cluster_variant_df[
            "hard_excluded_blacklisted_or_pathogenic"
        ]
    )

    renamed_cluster_variant_df[
        "discarded_poor_qc_missing_baseline_germline_evidence"
    ] = (
        require_baseline_germline_evidence
        & provisional_marker_candidate_mask
        & ~renamed_cluster_variant_df["has_baseline_germline_evidence"]
        & ~renamed_cluster_variant_df["is_loh_variant"].fillna(False)
    )
    renamed_cluster_variant_df[
        "discarded_poor_qc_non_snv_or_symbolic_allele"
    ] = (
        provisional_marker_candidate_mask
        & ~renamed_cluster_variant_df["is_simple_snv"]
        & ~bool(allow_non_snv_germline_markers)
    )

    # A baseline seed can be internally consistent yet still represent a
    # systematically biased locus rather than a usable germline-HET marker.
    # Restrict this exclusion to variants assigned to germline_het after the
    # baseline-preservation step, and require low between-label dispersion.
    # Preliminary LOH-cluster members are protected by default: their current
    # AF may be uniformly extreme because a real event is broad in this sample.
    _label_min = pd.to_numeric(
        renamed_cluster_variant_df.get(
            "label_median_min",
            pd.Series(np.nan, index=renamed_cluster_variant_df.index),
        ),
        errors="coerce",
    )
    _label_max = pd.to_numeric(
        renamed_cluster_variant_df.get(
            "label_median_max",
            pd.Series(np.nan, index=renamed_cluster_variant_df.index),
        ),
        errors="coerce",
    )
    _label_iqr = pd.to_numeric(
        renamed_cluster_variant_df.get(
            "label_median_iqr",
            pd.Series(np.nan, index=renamed_cluster_variant_df.index),
        ),
        errors="coerce",
    )
    _n_labels_with_data = pd.to_numeric(
        renamed_cluster_variant_df.get(
            "n_labels_with_data",
            pd.Series(0, index=renamed_cluster_variant_df.index),
        ),
        errors="coerce",
    ).fillna(0)
    _frac_labels_het_like = pd.to_numeric(
        renamed_cluster_variant_df.get(
            "frac_labels_het_like",
            pd.Series(np.nan, index=renamed_cluster_variant_df.index),
        ),
        errors="coerce",
    )
    _label_median_range = _label_max - _label_min

    renamed_cluster_variant_df["label_median_range"] = _label_median_range
    renamed_cluster_variant_df["systematic_skew_all_label_medians_low"] = (
        _label_max <= float(systematic_skew_low)
    ).fillna(False)
    renamed_cluster_variant_df["systematic_skew_all_label_medians_high"] = (
        _label_min >= float(systematic_skew_high)
    ).fillna(False)
    renamed_cluster_variant_df["systematic_skew_low_label_divergence"] = (
        (_label_iqr <= float(systematic_skew_max_label_median_iqr))
        & (_label_median_range <= float(systematic_skew_max_label_median_range))
    ).fillna(False)
    renamed_cluster_variant_df["systematic_skew_insufficient_het_like_labels"] = (
        _frac_labels_het_like < float(systematic_skew_min_frac_labels_het_like)
    ).fillna(False)
    renamed_cluster_variant_df["systematic_skew_preliminary_loh_protected"] = (
        bool(protect_preliminary_loh_variants_from_systematic_skew_filter)
        & renamed_cluster_variant_df["was_preliminary_loh_variant"].fillna(False)
    )

    _systematically_skewed_baseline_candidate = (
        renamed_cluster_variant_df["renamed_cluster"].eq(
            germline_renamed_cluster_name
        )
        & renamed_cluster_variant_df["has_independent_baseline_germline_evidence"].fillna(False)
        & renamed_cluster_variant_df["is_simple_snv"].fillna(False)
        & ~renamed_cluster_variant_df[
            "hard_excluded_blacklisted_or_pathogenic"
        ].fillna(False)
        & (_n_labels_with_data >= int(systematic_skew_min_labels_with_data))
        & (
            renamed_cluster_variant_df["systematic_skew_all_label_medians_low"]
            | renamed_cluster_variant_df["systematic_skew_all_label_medians_high"]
        )
        & renamed_cluster_variant_df["systematic_skew_low_label_divergence"]
        & renamed_cluster_variant_df["systematic_skew_insufficient_het_like_labels"]
        & ~renamed_cluster_variant_df["systematic_skew_preliminary_loh_protected"]
    )
    renamed_cluster_variant_df[
        "discarded_poor_qc_systematically_skewed_baseline_af"
    ] = (
        bool(exclude_systematically_skewed_baseline_markers)
        & _systematically_skewed_baseline_candidate
    )

    # ── Local linked-artifact block QC ────────────────────────────────────
    # A single low-AF rescued marker can be a genuine amplicon-biased HET.
    # In contrast, two or more rescued-only SNVs on the *same assay amplicon*
    # are poor-quality markers when they share a tightly concordant per-cell AF
    # trace AND each remains uniformly low/high with little label-to-label
    # median movement. That is technical AF skew without LOH: a genuine LOH
    # marker should either be in a designated LOH cluster or show materially
    # different AF medians between labels. The CNV extraction mapping is the
    # preferred block definition; genomic distance is only a fallback for
    # variants lacking an amplicon map.
    renamed_cluster_variant_df["local_linked_artifact_block_id"] = pd.Series(
        pd.NA, index=renamed_cluster_variant_df.index, dtype="object"
    )
    renamed_cluster_variant_df["local_linked_artifact_block_size"] = 0
    renamed_cluster_variant_df["local_linked_artifact_block_span_bp"] = np.nan
    renamed_cluster_variant_df["local_linked_artifact_grouping"] = pd.Series(
        pd.NA, index=renamed_cluster_variant_df.index, dtype="object"
    )
    renamed_cluster_variant_df["local_linked_artifact_amplicon"] = pd.Series(
        pd.NA, index=renamed_cluster_variant_df.index, dtype="object"
    )
    renamed_cluster_variant_df[
        "local_linked_artifact_same_amplicon"
    ] = False
    renamed_cluster_variant_df[
        "local_linked_artifact_same_amplicon_requirement_bypassed_no_mapping"
    ] = False
    renamed_cluster_variant_df[
        "local_linked_artifact_same_raw_af_cluster"
    ] = False
    renamed_cluster_variant_df[
        "local_linked_artifact_rescued_not_clean"
    ] = False
    renamed_cluster_variant_df[
        "local_linked_artifact_uniform_low_or_high"
    ] = False
    renamed_cluster_variant_df[
        "local_linked_artifact_low_label_median_range"
    ] = False
    renamed_cluster_variant_df[
        "local_linked_artifact_max_label_median_range"
    ] = np.nan
    renamed_cluster_variant_df[
        "local_linked_artifact_af_skewed_without_loh"
    ] = False
    renamed_cluster_variant_df[
        "local_linked_artifact_min_pairwise_shared_cells"
    ] = np.nan
    renamed_cluster_variant_df[
        "local_linked_artifact_min_pairwise_spearman"
    ] = np.nan
    renamed_cluster_variant_df[
        "local_linked_artifact_max_pairwise_median_abs_diff"
    ] = np.nan
    renamed_cluster_variant_df[
        "local_linked_artifact_profile_concordant"
    ] = False
    renamed_cluster_variant_df[
        "discarded_poor_qc_local_linked_artifact_block"
    ] = False

    _local_blocks_rows = []

    def _normalise_variant_amplicon_map(mapping):
        """Return a unique variant -> amplicon Series from flexible inputs.

        The expected input is ``out_all["df_mut_to_amplicon_dataframe"]``.
        A plain ``{variant: amplicon}`` dict or the companion
        ``out_all["variant_to_amp_dict"]`` is also accepted for convenience.
        Ambiguous one-to-many mappings are intentionally dropped rather than
        guessing an amplicon assignment.
        """
        if mapping is None:
            return pd.Series(dtype="object")

        if isinstance(mapping, pd.DataFrame):
            _map_df = mapping.copy()
        elif isinstance(mapping, pd.Series):
            _map_df = mapping.rename_axis("variant").reset_index(name="amplicon")
        elif isinstance(mapping, dict):
            if {"variant", "amplicon"}.issubset(mapping):
                _map_df = pd.DataFrame(mapping)
            else:
                _map_df = pd.DataFrame(
                    {"variant": list(mapping.keys()), "amplicon": list(mapping.values())}
                )
        else:
            try:
                _map_df = pd.DataFrame(mapping)
            except Exception:
                return pd.Series(dtype="object")

        if _map_df.empty:
            return pd.Series(dtype="object")

        _lower_to_col = {str(c).strip().lower(): c for c in _map_df.columns}
        _variant_col = _lower_to_col.get("variant", _lower_to_col.get("id"))
        _amplicon_col = _lower_to_col.get("amplicon")
        if _variant_col is None or _amplicon_col is None:
            raise ValueError(
                "df_mut_to_amplicon_dataframe must contain 'variant' and "
                "'amplicon' columns."
            )

        _map_df = _map_df[[_variant_col, _amplicon_col]].copy()
        _map_df.columns = ["variant", "amplicon"]
        _map_df["variant"] = _map_df["variant"].astype(str).str.strip()
        _map_df["amplicon"] = _map_df["amplicon"].where(
            _map_df["amplicon"].notna(), np.nan
        )
        _map_df["amplicon"] = _map_df["amplicon"].astype(str).str.strip()
        _map_df.loc[
            _map_df["amplicon"].isin(["", "nan", "None", "<NA>"]),
            "amplicon",
        ] = np.nan
        _map_df = _map_df.dropna(subset=["variant", "amplicon"])
        _map_df = _map_df[_map_df["variant"].ne("")]
        if _map_df.empty:
            return pd.Series(dtype="object")

        _n_amplicons = _map_df.groupby("variant")["amplicon"].nunique()
        _unambiguous = _n_amplicons[_n_amplicons.eq(1)].index
        _map_df = _map_df[_map_df["variant"].isin(_unambiguous)]
        _map_df = _map_df.drop_duplicates("variant", keep="first")
        return pd.Series(
            _map_df["amplicon"].to_numpy(),
            index=_map_df["variant"].to_numpy(),
            dtype="object",
        )

    def _normalise_amplicon_chrom(value):
        """Normalise chromosome labels for DNA/CNV amplicon matching."""
        if pd.isna(value):
            return pd.NA
        value = str(value).strip()
        if not value:
            return pd.NA
        value = value.lower().replace("chr", "")
        return value.upper() if value in {"x", "y", "m", "mt"} else value

    def _amplicon_mapping_from_sample(sample_obj):
        """Build a variant-to-amplicon mapping from a live Sample object.

        Preferred source
        ----------------
        ``Sample.dna.col_attrs['id']`` paired with
        ``Sample.dna.col_attrs['amplicon']``. This is the native Tapestri
        variant-to-amplicon annotation and is the source of truth when
        available.

        Fallback
        --------
        If DNA does not carry an amplicon column, map DNA variants to the
        intervals in ``Sample.cnv.col_attrs`` using chromosome + POS overlap
        with ``start_pos <= POS <= end_pos``. Ambiguous overlaps are not
        guessed; only one-to-one assignments are kept.
        """
        if sample_obj is None or not hasattr(sample_obj, "dna"):
            return pd.Series(dtype="object"), None

        def _attrs_to_df(assay):
            attrs = getattr(assay, "col_attrs", None)
            if attrs is None:
                return pd.DataFrame()
            try:
                out = pd.DataFrame(attrs).copy()
            except Exception:
                return pd.DataFrame()
            return out

        dna_attrs = _attrs_to_df(getattr(sample_obj, "dna", None))
        if dna_attrs.empty:
            return pd.Series(dtype="object"), None

        dna_lut = {str(c).strip().lower(): c for c in dna_attrs.columns}
        dna_variant_col = (
            dna_lut.get("id")
            or dna_lut.get("variant")
            or dna_lut.get("variant_id")
        )
        dna_amplicon_col = dna_lut.get("amplicon")

        # Direct DNA annotation is authoritative, but when CNV IDs are
        # present we validate that the stated amplicons actually exist there.
        if dna_variant_col is not None and dna_amplicon_col is not None:
            direct_map = _normalise_variant_amplicon_map(
                dna_attrs[[dna_variant_col, dna_amplicon_col]].rename(
                    columns={
                        dna_variant_col: "variant",
                        dna_amplicon_col: "amplicon",
                    }
                )
            )

            cnv_attrs = _attrs_to_df(getattr(sample_obj, "cnv", None))
            if not direct_map.empty and not cnv_attrs.empty:
                cnv_lut = {
                    str(c).strip().lower(): c for c in cnv_attrs.columns
                }
                cnv_id_col = cnv_lut.get("id") or cnv_lut.get("amplicon")
                if cnv_id_col is not None:
                    cnv_ids = set(
                        cnv_attrs[cnv_id_col].dropna().astype(str).str.strip()
                    )
                    direct_map = direct_map[
                        direct_map.astype(str).isin(cnv_ids)
                    ]

            if not direct_map.empty:
                return direct_map, "dna_col_attrs.variant_to_amplicon"

        # Derive variant-to-amplicon mapping by overlap against CNV intervals
        # only when direct DNA amplicon annotation is unavailable.
        if dna_variant_col is None:
            return pd.Series(dtype="object"), None

        cnv_attrs = _attrs_to_df(getattr(sample_obj, "cnv", None))
        if cnv_attrs.empty:
            return pd.Series(dtype="object"), None

        cnv_lut = {str(c).strip().lower(): c for c in cnv_attrs.columns}
        dna_chrom_col = (
            dna_lut.get("chrom")
            or dna_lut.get("chromosome")
            or dna_lut.get("chr")
        )
        dna_pos_col = (
            dna_lut.get("pos")
            or dna_lut.get("position")
            or dna_lut.get("start_pos")
        )
        cnv_chrom_col = (
            cnv_lut.get("chrom")
            or cnv_lut.get("chromosome")
            or cnv_lut.get("chr")
        )
        cnv_start_col = (
            cnv_lut.get("start_pos")
            or cnv_lut.get("start")
            or cnv_lut.get("begin")
        )
        cnv_end_col = (
            cnv_lut.get("end_pos")
            or cnv_lut.get("end")
            or cnv_lut.get("stop")
        )
        cnv_id_col = cnv_lut.get("id") or cnv_lut.get("amplicon")

        required = [
            dna_chrom_col,
            dna_pos_col,
            cnv_chrom_col,
            cnv_start_col,
            cnv_end_col,
            cnv_id_col,
        ]
        if any(col is None for col in required):
            return pd.Series(dtype="object"), None

        dna_coords = dna_attrs[
            [dna_variant_col, dna_chrom_col, dna_pos_col]
        ].copy()
        dna_coords.columns = ["variant", "chrom", "pos"]
        dna_coords["variant"] = dna_coords["variant"].astype(str).str.strip()
        dna_coords["chrom"] = dna_coords["chrom"].map(
            _normalise_amplicon_chrom
        )
        dna_coords["pos"] = pd.to_numeric(dna_coords["pos"], errors="coerce")
        dna_coords = dna_coords.dropna(subset=["variant", "chrom", "pos"])

        cnv_intervals = cnv_attrs[
            [cnv_chrom_col, cnv_start_col, cnv_end_col, cnv_id_col]
        ].copy()
        cnv_intervals.columns = ["chrom", "start", "end", "amplicon"]
        cnv_intervals["chrom"] = cnv_intervals["chrom"].map(
            _normalise_amplicon_chrom
        )
        cnv_intervals["start"] = pd.to_numeric(
            cnv_intervals["start"], errors="coerce"
        )
        cnv_intervals["end"] = pd.to_numeric(
            cnv_intervals["end"], errors="coerce"
        )
        cnv_intervals["amplicon"] = cnv_intervals["amplicon"].astype(str).str.strip()
        cnv_intervals = cnv_intervals.dropna(
            subset=["chrom", "start", "end", "amplicon"]
        )
        cnv_intervals = cnv_intervals[
            cnv_intervals["amplicon"].ne("")
        ].copy()

        if dna_coords.empty or cnv_intervals.empty:
            return pd.Series(dtype="object"), None

        rows = []
        for chrom, variants_on_chrom in dna_coords.groupby("chrom", sort=False):
            intervals_on_chrom = cnv_intervals.loc[
                cnv_intervals["chrom"].eq(chrom)
            ]
            if intervals_on_chrom.empty:
                continue
            for rec in variants_on_chrom.itertuples(index=False):
                hits = intervals_on_chrom.loc[
                    (intervals_on_chrom["start"] <= rec.pos)
                    & (intervals_on_chrom["end"] >= rec.pos),
                    "amplicon",
                ].drop_duplicates()
                if hits.nunique() == 1:
                    rows.append({
                        "variant": rec.variant,
                        "amplicon": hits.iloc[0],
                    })

        interval_map = _normalise_variant_amplicon_map(pd.DataFrame(rows))
        if interval_map.empty:
            return pd.Series(dtype="object"), None
        return interval_map, "cnv_col_attrs.interval_overlap"

    def _resolve_background_variant_amplicon_mapping(explicit_mapping):
        """Resolve a usable variant-to-amplicon map without changing the
        notebook call.

        Resolution is deliberately validation-aware: an empty/stale dataframe
        or an object without usable ``variant``/``amplicon`` columns is skipped
        instead of blocking later valid sources such as ``out_all``.

        Resolution order
        ----------------
        1. Explicit ``df_mut_to_amplicon_dataframe`` function argument.
        2. Input dataframes that already carry ``variant`` and ``amplicon``.
        3. Direct notebook variables:
           ``df_mut_to_amplicon_all_dataframe`` /
           ``df_mut_to_amplicon_dataframe``.
        4. ``out_all`` outputs from ``CNV_Utils.extract_cnv_data``.
        5. Matching objects in caller stack frames.
        """

        import inspect
        import sys

        direct_names = (
            "df_mut_to_amplicon_all_dataframe",
            "df_mut_to_amplicon_dataframe",
        )
        sample_names = (
            "Sample",
            "Sample_all_vars",
            "sample",
            "sample_all_vars",
            "Sample_T_subset",
        )
        out_all_names = (
            "df_mut_to_amplicon_dataframe",
            "variant_to_amp_dict",
            "df_mut_to_amplicon",
        )

        def _is_usable(candidate):
            if candidate is None:
                return False
            try:
                return not _normalise_variant_amplicon_map(candidate).empty
            except Exception:
                return False

        def _from_namespace(namespace, namespace_name):
            if not isinstance(namespace, dict):
                return None, None

            for name in direct_names:
                candidate = namespace.get(name)
                if _is_usable(candidate):
                    return candidate, f"{namespace_name}:{name}"

            # A live Sample object is more robust than notebook-variable
            # discovery. Its DNA col_attrs usually already carries the native
            # variant-to-amplicon annotation; CNV intervals are used when it
            # does not.
            for name in sample_names:
                sample_obj = namespace.get(name)
                sample_map, sample_source = _amplicon_mapping_from_sample(
                    sample_obj
                )
                if not sample_map.empty:
                    return sample_map, f"{namespace_name}:{name}.{sample_source}"

            out_all_obj = namespace.get("out_all")
            if isinstance(out_all_obj, dict):
                for key in out_all_names:
                    candidate = out_all_obj.get(key)
                    if _is_usable(candidate):
                        return candidate, f"{namespace_name}:out_all[{key!r}]"

            return None, None

        # Explicit input has highest priority.
        if _is_usable(explicit_mapping):
            return explicit_mapping, "explicit_function_argument"

        # Some pipeline tables retain an amplicon column. Prefer that direct,
        # reproducible provenance over notebook namespace discovery.
        for candidate, source in (
            (all_variant_germline_df, "all_variant_germline_df"),
            (df, "df"),
            (variant_af_cluster_df, "variant_af_cluster_df"),
            (df_AF_all, "df_AF_all"),
        ):
            if _is_usable(candidate):
                return candidate, source

        # In Jupyter, user variables normally live in __main__ or IPython's
        # user namespace. Check both, then walk caller frames as a fallback.
        namespaces = []
        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            namespaces.append((getattr(main_mod, "__dict__", None), "__main__"))

        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                namespaces.append((getattr(ip, "user_ns", None), "IPython.user_ns"))
        except Exception:
            pass

        for namespace, namespace_name in namespaces:
            candidate, source = _from_namespace(namespace, namespace_name)
            if candidate is not None:
                return candidate, source

        frame = inspect.currentframe()
        try:
            frame = frame.f_back
            max_frames = 24
            frame_i = 0
            while frame is not None and frame_i < max_frames:
                for namespace, scope in (
                    (frame.f_locals, "locals"),
                    (frame.f_globals, "globals"),
                ):
                    candidate, source = _from_namespace(
                        namespace,
                        f"caller_frame_{frame_i}.{scope}",
                    )
                    if candidate is not None:
                        return candidate, source
                frame = frame.f_back
                frame_i += 1
        finally:
            del frame

        return None, "unavailable"

    _resolved_amplicon_mapping, _amplicon_mapping_source = (
        _resolve_background_variant_amplicon_mapping(
            df_mut_to_amplicon_dataframe
        )
    )
    _variant_to_amplicon = _normalise_variant_amplicon_map(
        _resolved_amplicon_mapping
    )

    if bool(exclude_local_linked_artifact_blocks):
        _coords = renamed_cluster_variant_df["variant"].str.extract(
            r"^(chr[^:]+):(\d+):"
        )
        _coords.columns = ["_local_artifact_chrom", "_local_artifact_pos"]
        _coords["_local_artifact_pos"] = pd.to_numeric(
            _coords["_local_artifact_pos"], errors="coerce"
        )
        _local_source_text = (
            renamed_cluster_variant_df[
                "independent_baseline_germline_evidence_sources"
            ]
            .fillna("")
            .astype(str)
            .str.lower()
        )
        _local_has_clean_seed = _local_source_text.str.contains(
            r"(?:calibrated_)?clean_het_seed|clean_germline_het_seed",
            regex=True,
            na=False,
        )
        _local_has_rescued_seed = _local_source_text.str.contains(
            r"(?:calibrated_)?rescued_het_seed|rescued_germline_het_seed",
            regex=True,
            na=False,
        )
        _local_rescued_not_clean = (
            _local_has_rescued_seed & ~_local_has_clean_seed
        )

        _local_candidate = (
            renamed_cluster_variant_df["renamed_cluster"].eq(
                germline_renamed_cluster_name
            )
            & renamed_cluster_variant_df[
                "has_independent_baseline_germline_evidence"
            ].fillna(False)
            & renamed_cluster_variant_df["is_simple_snv"].fillna(False)
            & ~renamed_cluster_variant_df[
                "was_preliminary_loh_variant"
            ].fillna(False)
            & ~renamed_cluster_variant_df[
                "is_loh_variant"
            ].fillna(False)
            & ~renamed_cluster_variant_df[
                "hard_excluded_blacklisted_or_pathogenic"
            ].fillna(False)
        )
        if bool(local_artifact_require_rescued_not_clean):
            _local_candidate = _local_candidate & _local_rescued_not_clean

        _local_candidates = renamed_cluster_variant_df.loc[
            _local_candidate
        ].copy()
        _local_candidates = _local_candidates.join(_coords)
        _local_candidates["_local_artifact_amplicon"] = (
            _local_candidates["variant"].astype(str).map(_variant_to_amplicon)
        )

        def _evaluate_local_block(
            block_indices,
            *,
            grouping,
            amplicon=None,
        ):
            """Score one same-amplicon or positional candidate block."""
            if len(block_indices) < int(local_artifact_min_block_variants):
                return False

            _rows = renamed_cluster_variant_df.loc[block_indices].copy()
            _rows = _rows.join(_coords)
            _rows["_local_artifact_amplicon"] = (
                _rows["variant"].astype(str).map(_variant_to_amplicon)
            )
            _rows = _rows.sort_values(
                "_local_artifact_pos", na_position="last"
            )
            _vars = _rows["variant"].astype(str).tolist()
            _positions = pd.to_numeric(
                _rows["_local_artifact_pos"], errors="coerce"
            ).dropna().astype(int).tolist()
            _block_size = len(_vars)
            _block_span = (
                int(max(_positions) - min(_positions))
                if _positions else np.nan
            )
            _chroms = _rows["_local_artifact_chrom"].dropna().astype(str).unique()
            _chrom = _chroms[0] if len(_chroms) == 1 else "multiple"
            _block_start = int(min(_positions)) if _positions else np.nan
            _block_end = int(max(_positions)) if _positions else np.nan
            _amplicon = str(amplicon) if amplicon is not None else pd.NA
            if grouping == "amplicon":
                _block_id = f"amplicon:{_amplicon}"
            else:
                _block_id = f"{_chrom}:{_block_start}-{_block_end}"

            _same_raw_cluster = (
                _rows["raw_af_cluster_norm"].nunique(dropna=False) == 1
            )
            _same_amplicon = (
                grouping == "amplicon"
                and _rows["_local_artifact_amplicon"].notna().all()
                and _rows["_local_artifact_amplicon"].nunique(dropna=False) == 1
            )
            _all_rescued_not_clean = bool(
                _local_rescued_not_clean.loc[_rows.index].all()
            )
            _block_label_min = _label_min.loc[_rows.index]
            _block_label_max = _label_max.loc[_rows.index]
            _uniform_low = bool(
                (_block_label_max <= float(local_artifact_low)).fillna(False).all()
            )
            _uniform_high = bool(
                (_block_label_min >= float(local_artifact_high)).fillna(False).all()
            )
            _uniform_direction = _uniform_low or _uniform_high
            _block_label_ranges = pd.to_numeric(
                _label_median_range.loc[_rows.index], errors="coerce"
            )
            _max_label_median_range = (
                float(_block_label_ranges.max())
                if _block_label_ranges.notna().any()
                else np.nan
            )
            _low_label_median_range = bool(
                (
                    _block_label_ranges
                    <= float(local_artifact_max_label_median_range)
                ).fillna(False).all()
            )
            _is_loh_block = bool(
                _rows["was_preliminary_loh_variant"].fillna(False).any()
                or _rows["is_loh_variant"].fillna(False).any()
            )
            _af_skewed_without_loh = bool(
                _uniform_direction
                and _low_label_median_range
                and not _is_loh_block
            )

            _min_shared = np.inf
            _min_spearman = np.inf
            _max_median_abs_diff = 0.0
            _profiles_ok = True

            for _i in range(_block_size):
                for _j in range(_i + 1, _block_size):
                    if _vars[_i] not in af_mat.columns or _vars[_j] not in af_mat.columns:
                        _profiles_ok = False
                        _min_shared = np.nan
                        _min_spearman = np.nan
                        continue
                    _a = pd.to_numeric(af_mat[_vars[_i]], errors="coerce")
                    _b = pd.to_numeric(af_mat[_vars[_j]], errors="coerce")
                    _joint = _a.notna() & _b.notna()
                    _n_joint = int(_joint.sum())
                    _min_shared = min(_min_shared, _n_joint)
                    if _n_joint < int(local_artifact_min_pairwise_shared_cells):
                        _profiles_ok = False
                        _min_spearman = np.nan
                        continue
                    _a_joint = _a.loc[_joint]
                    _b_joint = _b.loc[_joint]
                    _rho = _a_joint.corr(_b_joint, method="spearman")
                    _median_abs_diff = float(
                        np.nanmedian(
                            np.abs(_a_joint.to_numpy() - _b_joint.to_numpy())
                        )
                    )
                    if not np.isfinite(_rho):
                        _profiles_ok = False
                    else:
                        _min_spearman = min(_min_spearman, float(_rho))
                        if _rho < float(local_artifact_min_pairwise_spearman):
                            _profiles_ok = False
                    _max_median_abs_diff = max(
                        _max_median_abs_diff, _median_abs_diff
                    )
                    if _median_abs_diff > float(
                        local_artifact_max_pairwise_median_abs_diff
                    ):
                        _profiles_ok = False

            if not np.isfinite(_min_shared):
                _min_shared = np.nan
            if not np.isfinite(_min_spearman):
                _min_spearman = np.nan

            _same_cluster_ok = (
                _same_raw_cluster
                if bool(local_artifact_require_same_raw_af_cluster)
                else True
            )
            _mapping_unavailable = _variant_to_amplicon.empty
            _same_amplicon_requirement_bypassed = bool(
                bool(local_artifact_require_same_amplicon)
                and bool(local_artifact_allow_position_fallback_without_amplicon_mapping)
                and _mapping_unavailable
                and grouping == "position"
            )
            _same_amplicon_ok = (
                True
                if _same_amplicon_requirement_bypassed
                else (
                    _same_amplicon
                    if bool(local_artifact_require_same_amplicon)
                    else True
                )
            )
            _label_range_ok = (
                _low_label_median_range
                if bool(local_artifact_require_low_label_median_range)
                else True
            )
            # The primary technical-artifact definition is assay-level:
            # >=2 rescued-only simple SNVs on the same amplicon, all uniformly
            # low/high across labels with a small label-median range, and no
            # evidence of LOH. Pairwise cell-level AF concordance is retained
            # for audit but is optional because depth/dropout can decorrelate
            # two manifestations of the same problematic amplicon.
            _profile_gate_ok = (
                _profiles_ok
                if bool(local_artifact_require_profile_concordance)
                else True
            )
            _discard = bool(
                _same_amplicon_ok
                and _same_cluster_ok
                and _all_rescued_not_clean
                and _af_skewed_without_loh
                and _label_range_ok
                and _profile_gate_ok
            )

            _target_index = _rows.index
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_block_id"
            ] = _block_id
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_block_size"
            ] = _block_size
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_block_span_bp"
            ] = _block_span
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_grouping"
            ] = grouping
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_amplicon"
            ] = _amplicon
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_same_amplicon"
            ] = bool(_same_amplicon)
            renamed_cluster_variant_df.loc[
                _target_index,
                "local_linked_artifact_same_amplicon_requirement_bypassed_no_mapping",
            ] = bool(_same_amplicon_requirement_bypassed)
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_same_raw_af_cluster"
            ] = bool(_same_raw_cluster)
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_rescued_not_clean"
            ] = bool(_all_rescued_not_clean)
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_uniform_low_or_high"
            ] = bool(_uniform_direction)
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_low_label_median_range"
            ] = bool(_low_label_median_range)
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_max_label_median_range"
            ] = _max_label_median_range
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_af_skewed_without_loh"
            ] = bool(_af_skewed_without_loh)
            renamed_cluster_variant_df.loc[
                _target_index,
                "local_linked_artifact_min_pairwise_shared_cells",
            ] = _min_shared
            renamed_cluster_variant_df.loc[
                _target_index,
                "local_linked_artifact_min_pairwise_spearman",
            ] = _min_spearman
            renamed_cluster_variant_df.loc[
                _target_index,
                "local_linked_artifact_max_pairwise_median_abs_diff",
            ] = _max_median_abs_diff
            renamed_cluster_variant_df.loc[
                _target_index, "local_linked_artifact_profile_concordant"
            ] = bool(_profiles_ok)
            if _discard:
                renamed_cluster_variant_df.loc[
                    _target_index,
                    "discarded_poor_qc_local_linked_artifact_block",
                ] = True

            _local_blocks_rows.append({
                "local_linked_artifact_block_id": _block_id,
                "grouping": grouping,
                "amplicon": _amplicon,
                "chromosome": _chrom,
                "block_start": _block_start,
                "block_end": _block_end,
                "block_span_bp": _block_span,
                "block_size": _block_size,
                "variants": ";".join(_vars),
                "same_amplicon": bool(_same_amplicon),
                "same_amplicon_requirement_bypassed_no_mapping": bool(
                    _same_amplicon_requirement_bypassed
                ),
                "same_raw_af_cluster": bool(_same_raw_cluster),
                "rescued_not_clean": bool(_all_rescued_not_clean),
                "uniform_low_or_high": bool(_uniform_direction),
                "low_label_median_range": bool(_low_label_median_range),
                "max_label_median_range": _max_label_median_range,
                "af_skewed_without_loh": bool(_af_skewed_without_loh),
                "is_loh_block": bool(_is_loh_block),
                "direction": (
                    "low" if _uniform_low else "high" if _uniform_high else "mixed"
                ),
                "min_pairwise_shared_cells": _min_shared,
                "min_pairwise_spearman": _min_spearman,
                "max_pairwise_median_abs_diff": _max_median_abs_diff,
                "profile_concordant": bool(_profiles_ok),
                "profile_concordance_required": bool(
                    local_artifact_require_profile_concordance
                ),
                "discarded_poor_qc": _discard,
            })
            return _discard

        _amplicon_evaluated_idx = set()
        if (
            bool(local_artifact_use_amplicon_blocks)
            and not _variant_to_amplicon.empty
            and not _local_candidates.empty
        ):
            _mapped_candidates = _local_candidates.dropna(
                subset=["_local_artifact_amplicon"]
            ).copy()
            for _amplicon, _sub in _mapped_candidates.groupby(
                "_local_artifact_amplicon", sort=False
            ):
                _idx = _sub.index.tolist()
                _full_amplicon_block_discarded = _evaluate_local_block(
                    _idx,
                    grouping="amplicon",
                    amplicon=_amplicon,
                )

                # A single amplicon can contain more than one independent
                # germline marker. If the full group is not concordant, probe
                # every pair so that a genuine two-SNV technical sub-block is
                # still detected without forcing unrelated variants into it.
                if (
                    not _full_amplicon_block_discarded
                    and len(_idx) > int(local_artifact_min_block_variants)
                ):
                    from itertools import combinations
                    for _pair in combinations(_idx, 2):
                        _evaluate_local_block(
                            list(_pair),
                            grouping="amplicon",
                            amplicon=_amplicon,
                        )

                _amplicon_evaluated_idx.update(_idx)

        # Optional legacy positional fallback. It is disabled by default:
        # nearby calls are not assumed to share an amplicon.
        if bool(local_artifact_fallback_to_position_blocks):
            _position_candidates = _local_candidates.loc[
                ~_local_candidates.index.isin(_amplicon_evaluated_idx)
            ].dropna(subset=["_local_artifact_chrom", "_local_artifact_pos"]).copy()
            _position_candidates["_local_artifact_pos"] = (
                _position_candidates["_local_artifact_pos"].astype(int)
            )

            for _chrom, _sub in _position_candidates.groupby(
                "_local_artifact_chrom", sort=False
            ):
                _sub = _sub.sort_values("_local_artifact_pos")
                _block_indices = []
                _block_start = None
                for _row_idx, _row in _sub.iterrows():
                    _pos = int(_row["_local_artifact_pos"])
                    if not _block_indices:
                        _block_indices = [_row_idx]
                        _block_start = _pos
                    elif _pos - _block_start <= int(local_artifact_max_block_span_bp):
                        _block_indices.append(_row_idx)
                    else:
                        _evaluate_local_block(
                            _block_indices,
                            grouping="position",
                            amplicon=None,
                        )
                        _block_indices = [_row_idx]
                        _block_start = _pos
                _evaluate_local_block(
                    _block_indices,
                    grouping="position",
                    amplicon=None,
                )

    local_linked_artifact_blocks_df = pd.DataFrame(_local_blocks_rows)
    if verbose:
        print(
            "  Local amplicon-artifact QC: "
            f"mapping_source={_amplicon_mapping_source}; "
            f"mapped_variants={int(renamed_cluster_variant_df['variant'].astype(str).isin(_variant_to_amplicon.index).sum())}; "
            f"position_fallback_without_mapping={bool(local_artifact_allow_position_fallback_without_amplicon_mapping and local_artifact_fallback_to_position_blocks and _variant_to_amplicon.empty)}; "
            f"candidate_blocks={len(_local_blocks_rows)}; "
            f"discarded_blocks={int(sum(bool(x.get('discarded_poor_qc', False)) for x in _local_blocks_rows))}"
        )
    if not local_linked_artifact_blocks_df.empty:
        local_linked_artifact_blocks_df = (
            local_linked_artifact_blocks_df
            .sort_values(["chromosome", "block_start", "block_end"])
            .reset_index(drop=True)
        )

    poor_qc_mask = (
        renamed_cluster_variant_df[
            "discarded_poor_qc_missing_baseline_germline_evidence"
        ]
        | renamed_cluster_variant_df[
            "discarded_poor_qc_non_snv_or_symbolic_allele"
        ]
        | renamed_cluster_variant_df[
            "discarded_poor_qc_systematically_skewed_baseline_af"
        ]
        | renamed_cluster_variant_df[
            "discarded_poor_qc_local_linked_artifact_block"
        ]
    )
    renamed_cluster_variant_df.loc[
        poor_qc_mask,
        "renamed_cluster",
    ] = discarded_poor_qc_renamed_cluster_name

    # Poor-QC candidates must not seed active LOH analysis even when their raw
    # cluster was provisionally designated LOH.
    renamed_cluster_variant_df.loc[poor_qc_mask, "is_loh_variant"] = False

    renamed_cluster_variant_df.loc[
        renamed_cluster_variant_df[
            "hard_excluded_blacklisted_or_pathogenic"
        ],
        "renamed_cluster",
    ] = excluded_renamed_cluster_name
    renamed_cluster_variant_df.loc[
        renamed_cluster_variant_df[
            "hard_excluded_blacklisted_or_pathogenic"
        ],
        "is_loh_variant",
    ] = False

    def _assignment_reason(row):
        if bool(row.get("hard_excluded_blacklisted_or_pathogenic", False)):
            return (
                "excluded_from_germline_het__"
                + str(
                    row.get(
                        "hard_exclusion_reason",
                        "blacklisted_or_pathogenic",
                    )
                )
            )
        if bool(row.get("discarded_poor_qc_non_snv_or_symbolic_allele", False)):
            return (
                "discarded_poor_qc__"
                + str(row.get("variant_allele_qc_reason", "non_snv_or_symbolic_allele"))
            )
        if bool(
            row.get(
                "discarded_poor_qc_missing_baseline_germline_evidence",
                False,
            )
        ):
            return (
                "discarded_poor_qc__marker_candidate_without_"
                "independent_baseline_germline_evidence"
            )
        if bool(
            row.get(
                "discarded_poor_qc_local_linked_artifact_block",
                False,
            )
        ):
            return "discarded_poor_qc__local_linked_artifact_block"
        if bool(
            row.get(
                "discarded_poor_qc_systematically_skewed_baseline_af",
                False,
            )
        ):
            return "discarded_poor_qc__systematically_skewed_baseline_af"
        if (
            bool(row.get("is_loh_variant", False))
            and not bool(row.get("has_independent_baseline_germline_evidence", False))
            and bool(row.get("has_input_filtered_germline_provenance", False))
        ):
            return (
                "preserved_validated_loh_marker_by_input_germline_provenance"
            )
        if bool(row.get("is_loh_variant", False)):
            return "preserved_validated_loh_marker_by_baseline_germline_evidence"
        if (
            preserve_baseline_germline_evidence
            and bool(row.get("has_baseline_germline_evidence", False))
        ):
            return "preserved_by_baseline_germline_evidence"
        if (
            bool(row.get("was_original_filtered_germline", False))
            and force_original_filtered_germline
        ):
            return "forced_original_filtered_germline_to_germline_het"
        if (
            bool(row.get("was_original_filtered_germline", False))
            and not force_original_filtered_germline
        ):
            return "original_filtered_germline_reassigned_by_AF_CLUSTER_RENAME_MAP"
        if pd.notna(row.get("renamed_cluster_from_dict", np.nan)):
            return "assigned_by_AF_CLUSTER_RENAME_MAP"
        return "default_raw_cluster_name"

    renamed_cluster_variant_df["renamed_cluster_assignment_reason"] = (
        renamed_cluster_variant_df.apply(_assignment_reason, axis=1)
    )

    new_filtered_germline_variants = sorted(
        renamed_cluster_variant_df.loc[
            renamed_cluster_variant_df["renamed_cluster"].eq(
                germline_renamed_cluster_name
            )
            & (
                renamed_cluster_variant_df["has_baseline_germline_evidence"]
                | renamed_cluster_variant_df["is_loh_variant"].fillna(False)
            )
            & (
                renamed_cluster_variant_df["is_simple_snv"]
                | bool(allow_non_snv_germline_markers)
            )
            & ~renamed_cluster_variant_df[
                "hard_excluded_blacklisted_or_pathogenic"
            ],
            "variant",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist(),
        key=_chr_pos_key,
    )
    renamed_cluster_variant_df[
        "is_filtered_germline_after_renamed_cluster_update"
    ] = renamed_cluster_variant_df["variant"].isin(
        set(new_filtered_germline_variants)
    )

    excluded_blacklisted_pathogenic_df = (
        renamed_cluster_variant_df.loc[
            renamed_cluster_variant_df[
                "hard_excluded_blacklisted_or_pathogenic"
            ]
        ]
        .copy()
        .sort_values(["hard_exclusion_reason", "raw_af_cluster", "variant"])
        .reset_index(drop=True)
    )
    discarded_poor_qc_df = (
        renamed_cluster_variant_df.loc[poor_qc_mask]
        .copy()
        .sort_values(
            ["was_preliminary_loh_variant", "raw_af_cluster", "variant"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    print("\nRenamed cluster reassignment")
    print("----------------------------")
    print(
        f"Original filtered_germline_variants:       "
        f"{len(original_filtered_set)}"
    )
    print(
        f"Independent baseline-supported markers:      "
        f"{int(renamed_cluster_variant_df['has_independent_baseline_germline_evidence'].sum())}"
    )
    print(
        f"Input-list provenance only:                 "
        f"{int((renamed_cluster_variant_df['has_input_filtered_germline_provenance'] & ~renamed_cluster_variant_df['has_independent_baseline_germline_evidence']).sum())}"
    )
    print(
        f"LOH-supported by input provenance:          "
        f"{int((renamed_cluster_variant_df['is_loh_variant'] & ~renamed_cluster_variant_df['has_independent_baseline_germline_evidence']).sum())}"
    )
    print(
        f"Discarded poor-QC marker candidates:        "
        f"{discarded_poor_qc_df.shape[0]}"
    )
    print(
        f"  ├─ missing independent evidence:          "
        f"{int(renamed_cluster_variant_df['discarded_poor_qc_missing_baseline_germline_evidence'].sum())}"
    )
    print(
        f"  ├─ non-SNV/symbolic alleles:              "
        f"{int(renamed_cluster_variant_df['discarded_poor_qc_non_snv_or_symbolic_allele'].sum())}"
    )
    print(
        f"  ├─ systematically skewed baseline AF:     "
        f"{int(renamed_cluster_variant_df['discarded_poor_qc_systematically_skewed_baseline_af'].sum())}"
    )
    print(
        f"  └─ local linked-artifact blocks:          "
        f"{int(renamed_cluster_variant_df['discarded_poor_qc_local_linked_artifact_block'].sum())}"
    )
    print(
        f"Hard-excluded blacklisted/pathogenic:       "
        f"{excluded_blacklisted_pathogenic_df.shape[0]}"
    )
    print(
        f"Updated filtered_germline_variants:         "
        f"{len(new_filtered_germline_variants)}"
    )
    print(
        f"Added to filtered_germline_variants:        "
        f"{len(set(new_filtered_germline_variants) - original_filtered_set)}"
    )
    print(
        f"Removed from original filtered germline:     "
        f"{len(original_filtered_set - set(new_filtered_germline_variants))}"
    )

    loh_variants = sorted(
        renamed_cluster_variant_df.loc[
            renamed_cluster_variant_df["is_loh_variant"], "variant"
        ]
        .astype(str)
        .drop_duplicates()
        .tolist(),
        key=_chr_pos_key,
    )
    loh_variant_df = (
        renamed_cluster_variant_df.loc[
            renamed_cluster_variant_df["is_loh_variant"]
        ]
        .copy()
        .sort_values(["raw_af_cluster", "variant"])
        .reset_index(drop=True)
    )
    print(
        "LOH-tagged validated germline variants "
        f"(raw clusters {sorted(loh_cluster_ids_norm, key=str)}): "
        f"{len(loh_variants)}"
    )

    renamed_cluster_summary_df = (
        renamed_cluster_variant_df.groupby("renamed_cluster")
        .agg(
            n_variants=("variant", "count"),
            n_original_filtered_germline=(
                "was_original_filtered_germline",
                "sum",
            ),
            n_baseline_germline_evidence=(
                "has_baseline_germline_evidence",
                "sum",
            ),
            n_independent_baseline_germline_evidence=(
                "has_independent_baseline_germline_evidence",
                "sum",
            ),
            n_loh_baseline_germline_evidence=(
                "has_loh_baseline_germline_evidence",
                "sum",
            ),
            n_input_list_provenance=(
                "has_input_filtered_germline_provenance",
                "sum",
            ),
            n_non_simple_snv=(
                "is_simple_snv",
                lambda x: int((~pd.Series(x).fillna(False).astype(bool)).sum()),
            ),
            n_filtered_germline_after_update=(
                "is_filtered_germline_after_renamed_cluster_update",
                "sum",
            ),
            n_loh_variants=("is_loh_variant", "sum"),
            n_preliminary_loh_variants=(
                "was_preliminary_loh_variant",
                "sum",
            ),
            n_discarded_poor_qc_missing_baseline=(
                "discarded_poor_qc_missing_baseline_germline_evidence",
                "sum",
            ),
            n_discarded_poor_qc_non_snv_or_symbolic=(
                "discarded_poor_qc_non_snv_or_symbolic_allele",
                "sum",
            ),
            n_discarded_poor_qc_systematically_skewed_baseline_af=(
                "discarded_poor_qc_systematically_skewed_baseline_af",
                "sum",
            ),
            n_discarded_poor_qc_local_linked_artifact_block=(
                "discarded_poor_qc_local_linked_artifact_block",
                "sum",
            ),
            n_hard_excluded_blacklisted_or_pathogenic=(
                "hard_excluded_blacklisted_or_pathogenic",
                "sum",
            ),
            n_hard_excluded_blacklisted=(
                "hard_excluded_blacklisted",
                "sum",
            ),
            n_hard_excluded_pathogenic=(
                "hard_excluded_pathogenic",
                "sum",
            ),
            raw_af_clusters=(
                "raw_af_cluster",
                lambda x: ",".join(
                    map(str, sorted(pd.Series(x).dropna().unique()))
                ),
            ),
            median_af_cluster=("median_af", "median"),
            median_mean_af=("mean_af", "median"),
            median_af_iqr=("af_iqr", "median"),
            median_frac_het_like=("frac_het_like", "median"),
            median_frac_clean_het=("frac_clean_het", "median"),
            median_frac_ref_like=("frac_ref_like", "median"),
            median_frac_alt_like=("frac_alt_like", "median"),
            median_frac_fixed_like=("frac_fixed_like", "median"),
            median_valid_frac=("valid_frac", "median"),
        )
        .reset_index()
        .sort_values(
            [
                "renamed_cluster",
                "n_filtered_germline_after_update",
                "n_variants",
            ],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )
    print("\nRenamed cluster summary")
    print("-----------------------")
    _display(renamed_cluster_summary_df)

    renamed_cluster_tables = {}
    if plot:
        clusters = sorted(
            renamed_cluster_variant_df["renamed_cluster"]
            .dropna()
            .unique()
            .tolist()
        )
        preferred_order = [
            germline_renamed_cluster_name,
            "germline_wt",
            "germline_hom",
            discarded_poor_qc_renamed_cluster_name,
            excluded_renamed_cluster_name,
        ]
        renamed_cluster_order = [
            c for c in preferred_order if c in clusters
        ] + [c for c in clusters if c not in preferred_order]
        label_order = labels.value_counts().index.tolist()
        if "WT" in label_order:
            label_order = ["WT"] + [x for x in label_order if x != "WT"]

        for renamed_cluster in renamed_cluster_order:
            plot_df = renamed_cluster_variant_df.loc[
                renamed_cluster_variant_df["renamed_cluster"].eq(
                    renamed_cluster
                )
            ].copy()
            if plot_df.empty:
                continue

            plot_df["_chr_sort"] = plot_df["variant"].map(
                lambda v: _chr_pos_key(v)[0]
            )
            plot_df["_pos_sort"] = plot_df["variant"].map(
                lambda v: _chr_pos_key(v)[1]
            )
            plot_df = plot_df.sort_values(
                [
                    "was_original_filtered_germline",
                    "has_independent_baseline_germline_evidence",
                    "has_loh_baseline_germline_evidence",
                    "has_baseline_germline_evidence",
                    "is_loh_variant",
                    "is_simple_snv",
                    "is_filtered_germline_after_renamed_cluster_update",
                    "discarded_poor_qc_missing_baseline_germline_evidence",
                    "discarded_poor_qc_non_snv_or_symbolic_allele",
                    "discarded_poor_qc_systematically_skewed_baseline_af",
                    "discarded_poor_qc_local_linked_artifact_block",
                    "hard_excluded_blacklisted_or_pathogenic",
                    "_chr_sort",
                    "_pos_sort",
                    "variant",
                ],
                ascending=[
                    False, False, False, False, False, False, False,
                    True, True, True, True, True, True, True, True,
                ],
            ).reset_index(drop=True)
            plot_df.insert(
                0,
                "renamed_cluster_x_index",
                np.arange(1, plot_df.shape[0] + 1),
            )
            plot_vars = plot_df["variant"].astype(str).tolist()
            af = af_mat[plot_vars].apply(pd.to_numeric, errors="coerce")

            cell_order = []
            for lab in label_order:
                idx = labels[labels.eq(lab)].index
                cell_order += (
                    af.loc[idx].mean(axis=1, skipna=True).sort_values().index.tolist()
                )
            if not cell_order:
                cell_order = af.mean(axis=1, skipna=True).sort_values().index.tolist()
            af = af.loc[cell_order, plot_vars]

            n_cells, n_vars = af.shape
            fig_width = max(8, min(32, 0.16 * n_vars + 6))
            fig_height = max(6, min(45, 0.045 * n_cells + 4))
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            im = ax.imshow(
                af.values,
                aspect="auto",
                interpolation="nearest",
                vmin=0,
                vmax=100,
                cmap="viridis",
            )
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label(
                "AF (%)"
            )
            if n_vars <= max_xticks:
                tick_positions = np.arange(n_vars)
            else:
                tick_positions = np.unique(
                    np.linspace(0, n_vars - 1, max_xticks)
                    .round()
                    .astype(int)
                )
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions + 1, fontsize=8)
            ax.set_yticks([])
            ax.tick_params(axis="both", which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

            n_original_fg = int(plot_df["was_original_filtered_germline"].sum())
            n_baseline = int(plot_df["has_baseline_germline_evidence"].sum())
            n_after_fg = int(
                plot_df["is_filtered_germline_after_renamed_cluster_update"].sum()
            )
            n_discarded_missing_baseline = int(
                plot_df[
                    "discarded_poor_qc_missing_baseline_germline_evidence"
                ].sum()
            )
            n_discarded_non_snv = int(
                plot_df[
                    "discarded_poor_qc_non_snv_or_symbolic_allele"
                ].sum()
            )
            n_discarded_skew = int(
                plot_df[
                    "discarded_poor_qc_systematically_skewed_baseline_af"
                ].sum()
            )
            n_discarded_local_block = int(
                plot_df[
                    "discarded_poor_qc_local_linked_artifact_block"
                ].sum()
            )
            n_discarded = (
                n_discarded_missing_baseline
                + n_discarded_non_snv
                + n_discarded_skew
                + n_discarded_local_block
            )
            n_hard_excluded = int(
                plot_df["hard_excluded_blacklisted_or_pathogenic"].sum()
            )
            n_loh = int(plot_df["is_loh_variant"].sum())
            prefix = f"{patient_id}: " if patient_id is not None else ""
            ax.set_xlabel(
                f"Variant index within renamed cluster (n={n_vars})"
            )
            ax.set_ylabel(f"Cells ordered by label (n={n_cells})")
            ax.set_title(
                f"{prefix}renamed cluster = {renamed_cluster} | "
                f"original_filtered={n_original_fg}, baseline={n_baseline}, "
                f"filtered_after_update={n_after_fg}, loh={n_loh}, "
                f"discarded_poor_qc={n_discarded} "
                f"(missing={n_discarded_missing_baseline}, "
                f"non_snv={n_discarded_non_snv}, skew={n_discarded_skew}, "
                f"local_block={n_discarded_local_block}), "
                f"hard_excluded={n_hard_excluded}",
                fontsize=12,
            )
            plt.tight_layout()
            plt.show()

            plot_df = plot_df.drop(
                columns=["_chr_sort", "_pos_sort"],
                errors="ignore",
            )
            display_cols = [
                "renamed_cluster_x_index",
                "variant",
                "renamed_cluster",
                "raw_af_cluster",
                "is_loh_variant",
                "was_preliminary_loh_variant",
                "was_original_filtered_germline",
                "has_input_filtered_germline_provenance",
                "has_any_upstream_germline_support",
                "has_independent_baseline_germline_evidence",
                "independent_baseline_germline_evidence_sources",
                "has_loh_baseline_germline_evidence",
                "has_baseline_germline_evidence",
                "baseline_germline_evidence_sources",
                "is_simple_snv",
                "variant_allele_qc_reason",
                "discarded_poor_qc_missing_baseline_germline_evidence",
                "discarded_poor_qc_non_snv_or_symbolic_allele",
                "discarded_poor_qc_systematically_skewed_baseline_af",
                "discarded_poor_qc_local_linked_artifact_block",
                "local_linked_artifact_block_id",
                "local_linked_artifact_block_size",
                "local_linked_artifact_same_amplicon_requirement_bypassed_no_mapping",
                "local_linked_artifact_block_span_bp",
                "local_linked_artifact_same_raw_af_cluster",
                "local_linked_artifact_rescued_not_clean",
                "local_linked_artifact_uniform_low_or_high",
                "local_linked_artifact_low_label_median_range",
                "local_linked_artifact_max_label_median_range",
                "local_linked_artifact_af_skewed_without_loh",
                "local_linked_artifact_min_pairwise_shared_cells",
                "local_linked_artifact_min_pairwise_spearman",
                "local_linked_artifact_max_pairwise_median_abs_diff",
                "local_linked_artifact_profile_concordant",
                "systematic_skew_all_label_medians_low",
                "systematic_skew_all_label_medians_high",
                "systematic_skew_low_label_divergence",
                "systematic_skew_insufficient_het_like_labels",
                "systematic_skew_preliminary_loh_protected",
                "label_median_range",
                "is_filtered_germline_after_renamed_cluster_update",
                "hard_excluded_blacklisted_or_pathogenic",
                "hard_exclusion_reason",
                "hard_exclusion_sources",
                "renamed_cluster_assignment_reason",
                "median_af",
                "mean_af",
                "af_iqr",
                "af_std",
                "frac_het_like",
                "frac_clean_het",
                "frac_ref_like",
                "frac_alt_like",
                "frac_fixed_like",
                "n_valid",
                "valid_frac",
                "n_labels_with_data",
                "frac_labels_het_like",
                "label_median_min",
                "label_median_max",
                "label_median_iqr",
            ]
            table = plot_df[
                [c for c in display_cols if c in plot_df.columns]
            ].copy()
            if include_variant_tables:
                print(
                    f"\nVariant table for renamed cluster: {renamed_cluster}"
                )
                _display(table)
            renamed_cluster_tables[renamed_cluster] = table

    updated_filtered_germline_df = (
        renamed_cluster_variant_df.loc[
            renamed_cluster_variant_df["variant"].isin(
                set(new_filtered_germline_variants)
            )
        ]
        .copy()
        .sort_values(["renamed_cluster", "raw_af_cluster", "variant"])
        .reset_index(drop=True)
    )
    print("\nUpdated filtered_germline_variants")
    print("----------------------------------")
    print(
        f"filtered_germline_variants now contains "
        f"{len(new_filtered_germline_variants)} variants."
    )
    _display(updated_filtered_germline_df)

    return {
        "AF_CLUSTER_RENAME_MAP": rename_map,
        "force_original_filtered_germline": force_original_filtered_germline,
        "preserve_baseline_germline_evidence": (
            preserve_baseline_germline_evidence
        ),
        "require_baseline_germline_evidence": (
            require_baseline_germline_evidence
        ),
        "allow_input_filtered_germline_as_baseline_evidence": (
            allow_input_filtered_germline_as_baseline_evidence
        ),
        "allow_input_filtered_germline_as_loh_evidence": (
            allow_input_filtered_germline_as_loh_evidence
        ),
        "allow_non_snv_germline_markers": allow_non_snv_germline_markers,
        "exclude_systematically_skewed_baseline_markers": (
            exclude_systematically_skewed_baseline_markers
        ),
        "exclude_local_linked_artifact_blocks": (
            exclude_local_linked_artifact_blocks
        ),
        "amplicon_mapping_source": _amplicon_mapping_source,
        "n_variants_with_amplicon_mapping": int(_variant_to_amplicon.size),
        "local_linked_artifact_parameters": {
            "min_block_variants": int(local_artifact_min_block_variants),
            "max_block_span_bp": int(local_artifact_max_block_span_bp),
            "require_same_raw_af_cluster": bool(
                local_artifact_require_same_raw_af_cluster
            ),
            "low": float(local_artifact_low),
            "high": float(local_artifact_high),
            "require_low_label_median_range": bool(
                local_artifact_require_low_label_median_range
            ),
            "max_label_median_range": float(
                local_artifact_max_label_median_range
            ),
            "require_same_amplicon": bool(local_artifact_require_same_amplicon),
            "allow_position_fallback_without_amplicon_mapping": bool(
                local_artifact_allow_position_fallback_without_amplicon_mapping
            ),
            "min_pairwise_shared_cells": int(
                local_artifact_min_pairwise_shared_cells
            ),
            "min_pairwise_spearman": float(
                local_artifact_min_pairwise_spearman
            ),
            "max_pairwise_median_abs_diff": float(
                local_artifact_max_pairwise_median_abs_diff
            ),
            "require_rescued_not_clean": bool(
                local_artifact_require_rescued_not_clean
            ),
        },
        "systematic_skew_parameters": {
            "low": float(systematic_skew_low),
            "high": float(systematic_skew_high),
            "max_label_median_iqr": float(
                systematic_skew_max_label_median_iqr
            ),
            "max_label_median_range": float(
                systematic_skew_max_label_median_range
            ),
            "min_labels_with_data": int(
                systematic_skew_min_labels_with_data
            ),
            "min_frac_labels_het_like": float(
                systematic_skew_min_frac_labels_het_like
            ),
            "protect_preliminary_loh_variants": bool(
                protect_preliminary_loh_variants_from_systematic_skew_filter
            ),
        },
        "germline_renamed_cluster_name": germline_renamed_cluster_name,
        "discarded_poor_qc_renamed_cluster_name": (
            discarded_poor_qc_renamed_cluster_name
        ),
        "renamed_cluster_variant_df": renamed_cluster_variant_df,
        "renamed_cluster_summary_df": renamed_cluster_summary_df,
        "renamed_cluster_tables": renamed_cluster_tables,
        "filtered_germline_variants_before_renamed_cluster_update": (
            original_filtered
        ),
        "filtered_germline_variants": new_filtered_germline_variants,
        "updated_filtered_germline_df": updated_filtered_germline_df,
        "excluded_blacklisted_pathogenic_df": (
            excluded_blacklisted_pathogenic_df
        ),
        "discarded_poor_qc_df": discarded_poor_qc_df,
        "local_linked_artifact_blocks_df": (
            local_linked_artifact_blocks_df
        ),
        "loh_variants": loh_variants,
        "loh_variant_df": loh_variant_df,
        "loh_cluster_ids": sorted(loh_cluster_ids_norm, key=str),
    }

# =============================================================================
# Step2 barcode-level all-variant CNV/BAF audit pipeline helpers
# =============================================================================
# These helpers turn the post-step2 notebook chunks into reusable pipeline calls:
#   1. build WT-vs-Non_WT all-variant matrices from step2["cell_clone_labels"]
#   2. run all-variant CNV/BAF informativeness, audit against
#      all_variant_germline_df, and fold in final no-loss CNV markers.


def _layer_df_from_sample(sample, layer):
    """Return a cell x variant dataframe for a sample DNA layer."""
    df = sample.dna.get_attribute(layer, constraint="row+col")

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(
            df,
            index=np.asarray(sample.dna.row_attrs["barcode"]).astype(str),
            columns=np.asarray(sample.dna.col_attrs["id"]).astype(str),
        )

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _normalise_cell_clone_labels(cell_clone_labels):
    """Normalise step2['cell_clone_labels'] to a barcode-indexed Series."""
    if isinstance(cell_clone_labels, pd.DataFrame):
        if {"barcode", "clone_label"}.issubset(cell_clone_labels.columns):
            out = pd.Series(
                cell_clone_labels["clone_label"].astype(str).values,
                index=cell_clone_labels["barcode"].astype(str).values,
                name="clone_label",
            )
        else:
            raise ValueError(
                "step2['cell_clone_labels'] is a DataFrame but does not contain "
                "'barcode' and 'clone_label' columns."
            )
    elif isinstance(cell_clone_labels, dict):
        out = pd.Series(cell_clone_labels, name="clone_label").astype(str)
    elif isinstance(cell_clone_labels, pd.Series):
        out = cell_clone_labels.copy()
        out.index = out.index.astype(str)
        out = out.astype(str)
        out.name = "clone_label"
    else:
        raise TypeError(
            "step2['cell_clone_labels'] must be a pandas Series, DataFrame, or dict."
        )

    out.index = out.index.astype(str)
    out = out.astype(str)
    out.name = "clone_label"
    return out


def build_all_variant_matrices_from_step2_clone_labels(
    step2,
    Sample,
    *,
    Sample_backup=None,
    wt_label: str = "WT",
    collapse_non_wt: bool = False,
    min_wt_cells: int = 5,
    min_non_wt_cells: int = 5,
    min_clone_cells: int = 5,
    verbose: bool = True,
    show_clone_barcode_head: bool = True,
):
    """Build all-variant AF/DP/ALT/REF matrices using barcode-level clone labels
    produced by :func:`add_wt_concordance_and_finalize_germline`.

    By default each clone keeps its own label (WT vs TP53_Het, WT vs TP53_Hom,
    ...), so informativeness is tested per clone rather than against a pooled
    Non_WT group. Pooling het and hom clones averages out subclonal LOH and was
    masking real markers. Set collapse_non_wt=True for the old WT-vs-Non_WT mode.

    Parameters
    ----------
    step2 : dict
        Output from ``add_wt_concordance_and_finalize_germline``. Must contain
        ``cell_clone_labels``.
    Sample : Mosaic sample
        Current sample object.
    Sample_backup : Mosaic sample, optional
        Full-variant sample object. If omitted, ``Sample`` is used.
    wt_label : str
        Clone label treated as WT reference.
    collapse_non_wt : bool
        If True, collapse every non-WT clone into a single ``Non_WT`` group.
        If False (default), keep each clone label as its own test group.
    min_wt_cells, min_non_wt_cells : int
        Minimum WT and total non-WT cell counts required.
    min_clone_cells : int
        Per-clone floor; clones below this are flagged (CNV_Utils skips them).

    Returns
    -------
    dict
        Contains ``df_AF_all``, ``df_DP_all``, ``df_ALT_all``, ``df_REF_all``,
        ``df_cell_clone_labels``, ``cell_clone_labels``, ``clone_barcode_df``,
        ``step2_clone_labels_used``, and ``sample_for_all_variants``.
    """
    if not isinstance(step2, dict):
        raise TypeError("step2 must be the dictionary returned by add_wt_concordance_and_finalize_germline.")

    if "cell_clone_labels" not in step2 or step2["cell_clone_labels"] is None:
        raise RuntimeError(
            "step2 does not contain 'cell_clone_labels'. Reload germline.py "
            "and rerun add_wt_concordance_and_finalize_germline(...)."
        )

    cell_clone_labels = _normalise_cell_clone_labels(step2["cell_clone_labels"])
    clone_barcode_df = step2.get("clone_barcode_df", None)
    step2_clone_labels_used = step2.get("step2_clone_labels_used", None)

    if verbose:
        print("Step2 clone labels used:")
        print(step2_clone_labels_used)
        print("\nBarcode-level clone labels from step2:")
        print(cell_clone_labels.value_counts(dropna=False).to_string())
        if clone_barcode_df is not None and show_clone_barcode_head:
            print("\nclone_barcode_df:")
            _display(clone_barcode_df.head())

    sample_for_all_variants = Sample_backup if Sample_backup is not None else Sample

    af_from_all = _layer_df_from_sample(sample_for_all_variants, "AF").astype(float)
    dp_from_all = _layer_df_from_sample(sample_for_all_variants, "DP").astype(float)

    variant_cols = af_from_all.columns.astype(str).tolist()
    shared_cells = af_from_all.index.astype(str).intersection(cell_clone_labels.index.astype(str))

    if len(shared_cells) == 0:
        raise RuntimeError(
            "No shared barcodes between sample_for_all_variants and step2['cell_clone_labels']."
        )

    raw_clone_labels = cell_clone_labels.reindex(shared_cells).astype(str)
    unique_raw_labels = raw_clone_labels.dropna().astype(str).unique().tolist()

    wt_like_labels = [lab for lab in unique_raw_labels if lab.strip().upper() == str(wt_label).upper()]
    if len(wt_like_labels) == 0:
        raise RuntimeError(
            "No WT label found among step2 barcode-level clone labels. "
            f"Available labels: {unique_raw_labels}"
        )
    wt_like_labels = set(wt_like_labels)

    is_wt = raw_clone_labels.isin(wt_like_labels)

    if collapse_non_wt:
        test_labels = pd.Series(
            np.where(is_wt, "WT", "Non_WT"),
            index=shared_cells,
            name="label",
        )
    else:
        # Each non-WT clone keeps its own label; only WT is normalised.
        test_labels = pd.Series(
            np.where(is_wt, "WT", raw_clone_labels.values),
            index=shared_cells,
            name="label",
        )

    df_AF_all = af_from_all.loc[shared_cells, variant_cols].copy()
    df_DP_all = dp_from_all.loc[shared_cells, variant_cols].copy()

    df_ALT_all = (
        df_DP_all * (df_AF_all / 100.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    df_REF_all = (
        df_DP_all - df_ALT_all
    ).clip(lower=0).replace([np.inf, -np.inf], np.nan).fillna(0)

    # CNV_Utils-compatible: one non-variant column only.
    df_AF_all.insert(0, "label", test_labels.values)

    df_cell_clone_labels = pd.DataFrame(
        {
            "barcode": shared_cells.astype(str),
            "clone_label_raw": raw_clone_labels.values,
            "label": test_labels.values,
        }
    ).set_index("barcode")

    label_counts = df_AF_all["label"].value_counts(dropna=False)
    n_wt = int(label_counts.get("WT", 0))
    non_wt_counts = label_counts.drop(labels=["WT"], errors="ignore")
    n_non_wt = int(non_wt_counts.sum())

    if verbose:
        print("\nBuilt all-variant matrices using step2 barcode-level clone labels")
        print("----------------------------------------------------------------")
        print(f"sample_for_all_variants: {type(sample_for_all_variants).__name__}")
        print(f"collapse_non_wt:         {collapse_non_wt}")
        print(f"df_AF_all:  {df_AF_all.shape}")
        print(f"df_DP_all:  {df_DP_all.shape}")
        print(f"df_ALT_all: {df_ALT_all.shape}")
        print(f"df_REF_all: {df_REF_all.shape}")
        print("\nRaw step2 clone label counts:")
        print(df_cell_clone_labels["clone_label_raw"].value_counts(dropna=False).to_string())
        print("\nTest-group label counts (WT vs each clone):")
        print(label_counts.to_string())
        small = non_wt_counts[non_wt_counts < min_clone_cells]
        if len(small):
            print(f"\n  ⚠ non-WT clones below min_clone_cells={min_clone_cells} "
                  "(CNV_Utils will skip these in testing):")
            print(small.to_string())

    if n_wt < min_wt_cells:
        raise RuntimeError(f"Fewer than {min_wt_cells} WT cells after step2 barcode relabelling.")
    if n_non_wt < min_non_wt_cells:
        raise RuntimeError(f"Fewer than {min_non_wt_cells} non-WT cells after step2 barcode relabelling.")
    if non_wt_counts[non_wt_counts >= min_clone_cells].empty:
        raise RuntimeError(
            f"No non-WT clone has at least min_clone_cells={min_clone_cells} cells; "
            "nothing to compare against WT."
        )

    return {
        "df_AF_all": df_AF_all,
        "df_DP_all": df_DP_all,
        "df_ALT_all": df_ALT_all,
        "df_REF_all": df_REF_all,
        "df_cell_clone_labels": df_cell_clone_labels,
        "cell_clone_labels": cell_clone_labels,
        "clone_barcode_df": clone_barcode_df,
        "step2_clone_labels_used": step2_clone_labels_used,
        "sample_for_all_variants": sample_for_all_variants,
    }

def _build_significant_summary_from_wilcoxon_results(
    wilcoxon_results_var,
    df_AF_all,
    *,
    p_threshold: float = 0.01,
    baf_deviation_threshold: float = 0.3,
    max_af_iqr: float = 0.3,
    allow_empty: bool = True,
    verbose: bool = True,
):
    """Resolve CNV_Utils output columns and build significant variant summaries.

    With allow_empty=True (default), a sample with no informative/significant
    variants returns an empty summary instead of raising, so the caller can fall
    back to the germline seed rather than failing the whole run.
    """
    variant_col = _first_existing_col(
        wilcoxon_results_var,
        ["variant", "variant_id", "id", "Variant", "Variant ID", "mutation", "locus"],
    )
    informative_col = _first_existing_col(
        wilcoxon_results_var,
        ["informative", "is_informative", "pass_informative", "pass_informativeness"],
    )
    significant_col = _first_existing_col(
        wilcoxon_results_var,
        ["significant", "is_significant", "pass_significant"],
    )
    p_col = _first_existing_col(
        wilcoxon_results_var,
        ["p_value", "pval", "p", "wilcoxon_p", "wilcoxon_pvalue", "p_adj", "q_value", "qval", "fdr"],
    )
    baf_col = _first_existing_col(
        wilcoxon_results_var,
        ["baf_deviation", "BAF_deviation", "abs_baf_deviation", "max_abs_baf_deviation", "delta_baf", "abs_delta_baf"],
    )
    iqr_col = _first_existing_col(
        wilcoxon_results_var,
        ["af_iqr", "AF_iqr", "baf_iqr", "BAF_iqr", "max_af_iqr", "variant_af_iqr"],
    )
    depth_col = _first_existing_col(
        wilcoxon_results_var,
        ["depth_ratio", "Depth_ratio", "median_depth_ratio", "clone_depth_ratio"],
    )
    event_col = _first_existing_col(
        wilcoxon_results_var,
        ["event_type", "Event_type", "event", "cnv_event"],
    )
    clone_col = _first_existing_col(
        wilcoxon_results_var,
        ["clone", "clone_label", "label", "test_label", "comparison_label", "group"],
    )

    if variant_col is None:
        raise ValueError(
            f"Could not identify variant column. Columns: {wilcoxon_results_var.columns.tolist()}"
        )

    if verbose:
        print("\nResolved Wilcoxon columns")
        print("-------------------------")
        print("variant:", variant_col)
        print("informative:", informative_col)
        print("significant:", significant_col)
        print("p:", p_col)
        print("baf:", baf_col)
        print("iqr:", iqr_col)
        print("depth:", depth_col)
        print("event:", event_col)
        print("clone:", clone_col)

    res = wilcoxon_results_var.copy()
    res[variant_col] = res[variant_col].astype(str)

    if informative_col is not None:
        sig_mask = _coerce_bool_like(res[informative_col]).fillna(False).astype(bool)
    elif significant_col is not None:
        sig_mask = _coerce_bool_like(res[significant_col]).fillna(False).astype(bool)
    else:
        if p_col is None:
            raise ValueError("No informative/significant column and no p-value column found.")
        sig_mask = pd.to_numeric(res[p_col], errors="coerce") <= p_threshold
        if baf_col is not None:
            sig_mask &= pd.to_numeric(res[baf_col], errors="coerce").abs() >= baf_deviation_threshold
        if iqr_col is not None:
            sig_mask &= pd.to_numeric(res[iqr_col], errors="coerce") <= max_af_iqr

    column_resolution = {
        "variant_col": variant_col,
        "informative_col": informative_col,
        "significant_col": significant_col,
        "p_col": p_col,
        "baf_col": baf_col,
        "iqr_col": iqr_col,
        "depth_col": depth_col,
        "event_col": event_col,
        "clone_col": clone_col,
    }

    sig_res = res.loc[sig_mask].copy()
    if sig_res.empty:
        msg = "No informative/significant variants found."
        if not allow_empty:
            raise RuntimeError(msg)
        if verbose:
            print(f"\n{msg} Proceeding with an empty significant set "
                  "(no CNV/BAF markers for this sample).")
        empty_summary = pd.DataFrame({
            "variant": pd.Series(dtype=str),
            "significant_informative": pd.Series(dtype=bool),
        })
        return {
            "sig_res": sig_res,
            "sig_variants_ordered": [],
            "significant_variant_summary": empty_summary,
            **column_resolution,
        }

    sig_variants = sig_res[variant_col].astype(str).drop_duplicates().tolist()
    sig_variants = [v for v in sig_variants if v in df_AF_all.columns]
    sig_variants_ordered = sorted(sig_variants, key=_chr_pos_key)

    if verbose:
        print(f"\nSignificant/informative comparisons: {sig_res.shape[0]}")
        print(f"Significant/informative unique variants present in df_AF_all: {len(sig_variants_ordered)}")

    significant_variant_summary = _build_significant_variant_summary(
        sig_res=sig_res,
        sig_variants_ordered=sig_variants_ordered,
        variant_col=variant_col,
        clone_col=clone_col,
        event_col=event_col,
        p_col=p_col,
        baf_col=baf_col,
        iqr_col=iqr_col,
        depth_col=depth_col,
    )

    return {
        "sig_res": sig_res,
        "sig_variants_ordered": sig_variants_ordered,
        "significant_variant_summary": significant_variant_summary,
        **column_resolution,
    }

def run_all_variant_cnv_baf_audit_pipeline(
    *,
    df_AF_all,
    df_DP_all,
    df_ALT_all,
    df_REF_all,
    all_variant_germline_df,
    CNV_Utils,
    DNA_Utils=None,
    wd: str = None,
    experiment: str = None,
    patient_id: str = None,
    cytobands=None,
    cytobands_path: str = None,
    output_dir: str = None,
    save_dir: str = GERMLINE_PATTERN_DIR,
    p_threshold: float = 0.01,
    baf_deviation_threshold: float = 0.3,
    depth_ratio_deletion: float = 0.7,
    depth_ratio_amplification: float = 1.3,
    max_af_iqr: float = 0.3,
    min_cells: int = 5,
    save_outputs: bool = True,
    apply_final_rules: bool = True,
    final_marker_parameters=None,
    error_on_empty: bool = False,
    verbose: bool = True,
    show_tables: bool = True,
):
    """Run the post-step2 all-variant CNV/BAF audit pipeline.

    When no CNV/BAF-informative variants are found (e.g. a het-only clone with
    no LOH to detect), the pipeline does not fail: with error_on_empty=False
    (default) it returns empty audit frames and falls back to the existing
    germline_het_seed with no CNV rescue. Set error_on_empty=True to hard-stop
    on such samples instead.
    """
    if "label" not in df_AF_all.columns:
        raise KeyError("df_AF_all must contain a 'label' column.")

    if verbose:
        print("Input label counts:")
        print(df_AF_all["label"].value_counts(dropna=False).to_string())

    if int(df_AF_all["label"].eq("WT").sum()) < min_cells:
        raise RuntimeError(f"Need at least {min_cells} WT cells in df_AF_all.")
    if int(df_AF_all["label"].ne("WT").sum()) < min_cells:
        raise RuntimeError(f"Need at least {min_cells} non-WT cells in df_AF_all.")

    df_ALT_all = df_ALT_all[[c for c in df_ALT_all.columns if c != "label"]].copy()
    df_REF_all = df_REF_all[[c for c in df_REF_all.columns if c != "label"]].copy()

    if cytobands is None:
        if cytobands_path is None:
            if wd is None:
                raise ValueError("Provide cytobands, cytobands_path, or wd.")
            cytobands_path = os.path.join(wd, "resources", "references", "cytobands.tsv")
        cytobands = pd.read_table(cytobands_path, index_col=[0])

    all_sample_backup_variants = [c for c in df_AF_all.columns.astype(str) if c != "label"]

    grouped_by_chr_all, grouped_by_chr_arm_all = CNV_Utils.group_informative_variants_by_chr_arm(
        filtered_germline_in_samples=all_sample_backup_variants,
        cytobands=cytobands,
        verbose=verbose,
    )

    if output_dir is None:
        if wd is not None and experiment is not None and patient_id is not None:
            output_dir = (
                f"{wd}/results/per_sample/{experiment}/{patient_id}/"
                "dna_cnv/clonotypes/allele_frequencies"
            )
        else:
            output_dir = save_dir

    wilcoxon_results_var_raw = CNV_Utils.test_variant_informativeness(
        df_AF=df_AF_all,
        df_ALT=df_ALT_all,
        df_REF=df_REF_all,
        grouped_by_chr_arm=grouped_by_chr_arm_all,
        output_dir=output_dir,
        p_threshold=p_threshold,
        baf_deviation_threshold=baf_deviation_threshold,
        depth_ratio_deletion=depth_ratio_deletion,
        depth_ratio_amplification=depth_ratio_amplification,
        max_af_iqr=max_af_iqr,
        min_cells=min_cells,
        verbose=verbose,
    )

    if isinstance(wilcoxon_results_var_raw, dict):
        frames = [
            d.assign(chr_arm=arm) if "chr_arm" not in d.columns else d.copy()
            for arm, d in wilcoxon_results_var_raw.items()
            if isinstance(d, pd.DataFrame) and not d.empty
        ]
        if len(frames) == 0:
            raise RuntimeError(
                "CNV_Utils.test_variant_informativeness returned no result tables. "
                "Check df_AF_all['label'] contains WT and non-WT labels."
            )
        wilcoxon_results_var = pd.concat(frames, ignore_index=True)
    else:
        wilcoxon_results_var = wilcoxon_results_var_raw.copy()

    if verbose:
        print("\nCombined Wilcoxon result table:")
        print(wilcoxon_results_var.shape)
    if show_tables:
        _display(wilcoxon_results_var.head())

    sig = _build_significant_summary_from_wilcoxon_results(
        wilcoxon_results_var,
        df_AF_all,
        p_threshold=p_threshold,
        baf_deviation_threshold=baf_deviation_threshold,
        max_af_iqr=max_af_iqr,
        allow_empty=not error_on_empty,
        verbose=verbose,
    )

    significant_variant_summary = sig["significant_variant_summary"]
    sig_variants_ordered = sig["sig_variants_ordered"]

    column_resolution = {k: sig[k] for k in [
        "variant_col", "informative_col", "significant_col", "p_col",
        "baf_col", "iqr_col", "depth_col", "event_col", "clone_col",
    ]}

    # No CNV/BAF-informative variant for this sample: don't fail, fall back to
    # the existing germline seed so downstream is unaffected.
    if len(sig_variants_ordered) == 0:
        if verbose:
            print("\nNo informative CNV/BAF markers for this sample; "
                  "falling back to germline_het_seed with no CNV rescue.")

        empty_audit = significant_variant_summary.copy()
        filtered_germline_variants = None
        germline_df_for_plot = None
        out_all_variant_germline_df = all_variant_germline_df

        if apply_final_rules:
            df = all_variant_germline_df.copy()
            df = df.drop(columns=["final_cnv_informative_marker_no_loss", "germline_tier"], errors="ignore")
            df["final_cnv_informative_marker_no_loss"] = False
            seed_col = _first_existing_col(df, [
                "germline_het_seed",
                "calibrated_usable_het_seed",
                "usable_germline_het_seed",
            ])
            clean_mask = (
                _coerce_bool_like(df[seed_col]).fillna(False).astype(bool)
                if seed_col is not None
                else pd.Series(False, index=df.index)
            )
            df["germline_tier"] = np.where(clean_mask, "clean_het", "none")
            filtered_germline_variants = df.loc[clean_mask, "variant"].astype(str).tolist()
            germline_df_for_plot = df.loc[clean_mask].copy()
            out_all_variant_germline_df = df

            if verbose:
                print(f"germline seed column used:           {seed_col}")
                print(f"clean germline_het_seed fallback:    {int(clean_mask.sum())}")
                print(f"filtered_germline_variants:          {len(filtered_germline_variants)}")

        return {
            "grouped_by_chr_all": grouped_by_chr_all,
            "grouped_by_chr_arm_all": grouped_by_chr_arm_all,
            "wilcoxon_results_var_raw": wilcoxon_results_var_raw,
            "wilcoxon_results_var": wilcoxon_results_var,
            "sig_res": sig["sig_res"],
            "sig_variants_ordered": sig_variants_ordered,
            "significant_variant_summary": significant_variant_summary,
            "audit_out": None,
            "audit_real": empty_audit,
            "missed_significant_in_all_variant_germline_df": empty_audit.copy(),
            "real_missed_variants_df": empty_audit.copy(),
            "cnv_out": None,
            "all_variant_germline_df": out_all_variant_germline_df,
            "filtered_germline_variants": filtered_germline_variants,
            "germline_df_for_plot": germline_df_for_plot,
            "column_resolution": column_resolution,
        }

    if verbose:
        print("\nSignificant variant summary")
        print("---------------------------")
    if show_tables:
        _display(significant_variant_summary)

    audit_fn = (
        DNA_Utils.check_significant_variants_against_all_variant_germline_df
        if DNA_Utils is not None and hasattr(DNA_Utils, "check_significant_variants_against_all_variant_germline_df")
        else check_significant_variants_against_all_variant_germline_df
    )

    audit_out = audit_fn(
        all_variant_germline_df=all_variant_germline_df,
        df_AF_all=df_AF_all,
        df_DP_all=df_DP_all,
        experiment=experiment,
        patient_id=patient_id,
        significant_variant_summary=significant_variant_summary,
        sig_variants_ordered=sig_variants_ordered,
        save_dir=save_dir,
        save_outputs=save_outputs,
        verbose=verbose,
    )

    audit_real = audit_out["audit_real"]
    missed_significant_in_all_variant_germline_df = audit_out["missed_significant_in_all_variant_germline_df"]
    real_missed_variants_df = audit_out["real_missed_variants_df"]

    if verbose:
        print("\nAudit outputs")
        print("-------------")
        print(f"audit_real:                                      {audit_real.shape}")
        print(f"missed_significant_in_all_variant_germline_df:  {missed_significant_in_all_variant_germline_df.shape}")
        print(f"real_missed_variants_df:                        {real_missed_variants_df.shape}")
    if show_tables:
        _display(missed_significant_in_all_variant_germline_df)

    cnv_out = None
    filtered_germline_variants = None
    germline_df_for_plot = None

    if apply_final_rules:
        final_fn = (
            DNA_Utils.apply_final_no_loss_cnv_marker_rules
            if DNA_Utils is not None and hasattr(DNA_Utils, "apply_final_no_loss_cnv_marker_rules")
            else apply_final_no_loss_cnv_marker_rules
        )
        cnv_out = final_fn(
            all_variant_germline_df,
            audit_real,
            parameters=final_marker_parameters or FINAL_NO_LOSS_CNV_MARKER_PARAMETERS,
            verbose=verbose,
        )
        all_variant_germline_df = cnv_out["all_variant_germline_df"]
        audit_real = cnv_out["audit_real"]
        filtered_germline_variants = cnv_out["filtered_germline_variants"]
        germline_df_for_plot = cnv_out["germline_df_for_plot"]

        if verbose:
            print("\nFinal CNV marker rule outputs")
            print("-----------------------------")
            print(f"all_variant_germline_df:      {all_variant_germline_df.shape}")
            print(f"audit_real:                   {audit_real.shape}")
            print(f"filtered_germline_variants:   {len(filtered_germline_variants)}")
            print(f"germline_df_for_plot:         {germline_df_for_plot.shape}")
        if show_tables and germline_df_for_plot is not None:
            _display(germline_df_for_plot.head())

    return {
        "grouped_by_chr_all": grouped_by_chr_all,
        "grouped_by_chr_arm_all": grouped_by_chr_arm_all,
        "wilcoxon_results_var_raw": wilcoxon_results_var_raw,
        "wilcoxon_results_var": wilcoxon_results_var,
        "sig_res": sig["sig_res"],
        "sig_variants_ordered": sig_variants_ordered,
        "significant_variant_summary": significant_variant_summary,
        "audit_out": audit_out,
        "audit_real": audit_real,
        "missed_significant_in_all_variant_germline_df": missed_significant_in_all_variant_germline_df,
        "real_missed_variants_df": real_missed_variants_df,
        "cnv_out": cnv_out,
        "all_variant_germline_df": all_variant_germline_df,
        "filtered_germline_variants": filtered_germline_variants,
        "germline_df_for_plot": germline_df_for_plot,
        "column_resolution": column_resolution,
    }

def run_step2_barcode_cnv_audit_pipeline(
    *,
    step2,
    Sample,
    all_variant_germline_df,
    CNV_Utils,
    DNA_Utils=None,
    Sample_backup=None,
    wd: str = None,
    experiment: str = None,
    patient_id: str = None,
    **kwargs,
):
    """Convenience wrapper combining the two post-step2 workflow steps.

    First builds all-variant matrices from ``step2['cell_clone_labels']`` (per
    clone by default), then runs the all-variant CNV/BAF audit pipeline.

    Extra keyword arguments are forwarded to
    :func:`run_all_variant_cnv_baf_audit_pipeline`.
    """
    matrix_kwargs = {}
    for key in [
        "wt_label",
        "collapse_non_wt",
        "min_wt_cells",
        "min_non_wt_cells",
        "min_clone_cells",
    ]:
        if key in kwargs:
            matrix_kwargs[key] = kwargs.pop(key)
    if "verbose" in kwargs:
        matrix_kwargs["verbose"] = kwargs["verbose"]

    matrices = build_all_variant_matrices_from_step2_clone_labels(
        step2,
        Sample,
        Sample_backup=Sample_backup,
        **matrix_kwargs,
    )

    audit = run_all_variant_cnv_baf_audit_pipeline(
        df_AF_all=matrices["df_AF_all"],
        df_DP_all=matrices["df_DP_all"],
        df_ALT_all=matrices["df_ALT_all"],
        df_REF_all=matrices["df_REF_all"],
        all_variant_germline_df=all_variant_germline_df,
        CNV_Utils=CNV_Utils,
        DNA_Utils=DNA_Utils,
        wd=wd,
        experiment=experiment,
        patient_id=patient_id,
        **kwargs,
    )

    out = dict(matrices)
    out.update(audit)
    out["matrices"] = matrices
    out["audit_pipeline"] = audit
    return out
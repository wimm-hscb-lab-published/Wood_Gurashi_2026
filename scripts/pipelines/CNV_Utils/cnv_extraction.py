import numpy as np
import pandas as pd
import os


def get_cnv_info(
    Sample,
    df_mut_to_amplicon: dict = None,
    desired_plotting_order: list = None,
    verbose: bool = True
) -> tuple:
    """
    Extract and process CNV information from Sample object.
    
    Creates three DataFrames:
    1. df_cnv: CNV ploidy data with cells as rows, amplicons as columns
    2. df_cnv_grouped: Median ploidy per clone, sorted by genomic order
    3. df_cnv_vars: CNV data mapped to variants (optional, requires df_mut_to_amplicon)
    
    Parameters
    ----------
    Sample : object
        Sample object with cnv attribute containing:
        - layers['ploidy']: CNV ploidy values
        - row_attrs['label']: Clone labels
        - col_attrs['id']: Probe IDs
        - col_attrs['CHROM']: Chromosome names
        - col_attrs['start_pos']: Genomic start positions
    df_mut_to_amplicon : dict, optional
        Dictionary mapping variants to amplicons with keys:
        - 'variant': list of variant IDs
        - 'amplicon': list of amplicon names
        If provided, creates df_cnv_vars with variant columns
    desired_plotting_order : list, optional
        Desired order for clone labels in df_cnv_grouped
        If None, uses all unique labels in natural order
    verbose : bool
        Print progress messages
        
    Returns
    -------
    df_cnv : pd.DataFrame
        Cell-level CNV data: rows = cells, columns = ['label'] + amplicons
        Sorted by genomic order
    df_cnv_grouped : pd.DataFrame
        Clone-level CNV data: rows = amplicons (genomic order), columns = clones
        Values are median ploidy per clone
    df_cnv_vars : pd.DataFrame or None
        Variant-level CNV data: rows = cells, columns = ['label'] + variants
        Only returned if df_mut_to_amplicon is provided
    cnv_info : pd.DataFrame
        Amplicon annotation sorted by genomic order
        Columns: 'id', 'CHROM', 'start_pos', etc.
    """
    
    import pandas as pd
    import numpy as np
    
    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()
    
    def chrom_key(col):
        """Sort key for chromosome names"""
        if col.name != 'CHROM':
            return col
        
        def chrom_to_int(chrom):
            chrom_str = str(chrom).upper().replace('CHR', '')
            if chrom_str == 'X':
                return 23
            elif chrom_str == 'Y':
                return 24
            elif chrom_str == 'M' or chrom_str == 'MT':
                return 25
            else:
                try:
                    return int(chrom_str)
                except ValueError:
                    return 999
        
        return col.map(chrom_to_int)
    
    vprint("\n" + "="*60)
    vprint("EXTRACTING CNV INFORMATION")
    vprint("="*60)
    
    # =========================================================================
    # STEP 1: Extract cell-level CNV data
    # =========================================================================
    vprint("\n  Step 1: Extracting cell-level CNV data...")
    
    # Get the 'ploidy' attribute
    df_cnv = Sample.cnv.get_attribute('ploidy', constraint='row')
    
    # Convert to DataFrame if necessary
    if not isinstance(df_cnv, pd.DataFrame):
        df_cnv = pd.DataFrame(df_cnv)
    
    # Add the 'label' column
    df_cnv['label'] = Sample.cnv.row_attrs["label"]
    
    # Reorder columns to have 'label' as the first column
    cols = list(df_cnv.columns.values)
    cols.insert(0, cols.pop(cols.index('label')))
    df_cnv = df_cnv[cols]
    
    # Drop duplicate columns and rows
    df_cnv = df_cnv.loc[:, ~df_cnv.columns.duplicated()]
    df_cnv = df_cnv.drop_duplicates()
    
    vprint(f"    ✓ Extracted CNV data: {df_cnv.shape[0]} cells × {df_cnv.shape[1]-1} amplicons")
    
    # =========================================================================
    # STEP 2: Build genomic annotation
    # =========================================================================
    vprint("\n  Step 2: Building genomic annotation...")
    
    # Build a tidy DataFrame from column attributes
    cnv_info = pd.DataFrame(Sample.cnv.col_attrs).drop_duplicates()
    
    # Ensure expected columns exist
    required_cols = {'CHROM', 'start_pos', 'id'}
    if not required_cols.issubset(cnv_info.columns):
        missing = required_cols - set(cnv_info.columns)
        raise KeyError(f"Missing required CNV columns: {missing}")
    
    # Make start_pos numeric
    cnv_info['start_pos'] = pd.to_numeric(cnv_info['start_pos'], errors='coerce')
    
    # Sort by CHROM then start_pos
    cnv_info_sorted = cnv_info.sort_values(
        by=['CHROM', 'start_pos'],
        key=chrom_key
    )
    
    vprint(f"    ✓ Sorted {len(cnv_info_sorted)} amplicons by genomic position")
    
    # Reorder df_cnv columns to genomic order
    ordered_columns = ['label'] + [col for col in cnv_info_sorted['id'] if col in df_cnv.columns]
    df_cnv = df_cnv[ordered_columns]
    
    vprint(f"    ✓ Reordered df_cnv columns to genomic order")
    
    # =========================================================================
    # STEP 3: Create clone-level grouped CNV data
    # =========================================================================
    vprint("\n  Step 3: Creating clone-level grouped CNV data...")
    
    # CRITICAL FIX: Use Sample.cnv column IDs directly (not cnv_info_sorted)
    # because Sample.cnv may have MORE columns than amplicons
    probe_ids = pd.Index(Sample.cnv.col_attrs['id'])
    df_cnv_grouped = pd.DataFrame(index=probe_ids)
    
    vprint(f"    Building grouped data with {len(probe_ids)} probes from Sample.cnv")
    
    # Determine clone order
    if desired_plotting_order is not None:
        # Only keep labels that exist in the data
        unique_labels = [lbl for lbl in desired_plotting_order 
                        if lbl in df_cnv['label'].unique()]
        vprint(f"    Using custom clone order: {len(unique_labels)} clones")
    else:
        # Use all unique labels
        unique_labels = sorted(df_cnv['label'].unique())
        vprint(f"    Using natural clone order: {len(unique_labels)} clones")
    
    # Fill columns with per-clone medians of ploidy
    labels_arr = np.asarray(Sample.cnv.row_attrs['label'])
    
    for clone in unique_labels:
        idx = np.where(labels_arr == clone)[0]
        if len(idx) == 0:
            vprint(f"      ⚠️  Warning: No cells found for clone '{clone}'")
            continue
        
        subset = Sample.cnv[idx, :]
        # Take median across cells (rows) - returns array matching Sample.cnv columns
        df_cnv_grouped[clone] = np.nanmedian(subset.layers['ploidy'], axis=0)
    
    # Remove duplicated rows
    df_cnv_grouped = df_cnv_grouped.drop_duplicates()
    
    # Reindex columns in desired order
    df_cnv_grouped = df_cnv_grouped.reindex(columns=unique_labels)
    
    # NOW reindex rows to match sorted genomic order (from cnv_info_sorted)
    # This filters to only the amplicons we want
    if df_cnv_grouped.index.has_duplicates:
        df_cnv_grouped = df_cnv_grouped.groupby(level=0).median()

    # 2) ensure target index is unique + ordered
    target_ids = pd.Index(cnv_info_sorted["id"]).drop_duplicates(keep="first")

    # 3) now reindex safely
    df_cnv_grouped = df_cnv_grouped.reindex(target_ids)
    
    vprint(f"    ✓ Created grouped CNV: {df_cnv_grouped.shape[0]} amplicons × {df_cnv_grouped.shape[1]} clones")
    
    # =========================================================================
    # STEP 4: Map variants to amplicons (optional)
    # =========================================================================
    df_cnv_vars = None
    
    if df_mut_to_amplicon is not None:
        vprint("\n  Step 4: Mapping variants to amplicons...")
        
        # Turn mapping into a DataFrame
        map_df = pd.DataFrame(df_mut_to_amplicon)  # columns: 'variant', 'amplicon'
        
        vprint(f"    Mapping: {len(map_df)} variants to amplicons")
        
        # Which amplicons are actually present in df_cnv
        amplicons_in_cnv = set(df_cnv.columns)
        
        # Report missing amplicons
        missing_amps = set(map_df["amplicon"]) - amplicons_in_cnv
        if missing_amps:
            vprint(f"    ⚠️  Warning: {len(missing_amps)} amplicons from mapping "
                  f"not found in df_cnv (will be filled with NaN)")
            if verbose and len(missing_amps) <= 10:
                for amp in sorted(missing_amps):
                    vprint(f"        • {amp}")
        
        # Treat all columns in df_cnv that are NOT amplicons as metadata
        meta_cols = [c for c in df_cnv.columns if c not in set(map_df["amplicon"])]
        
        # Start df_cnv_vars from metadata columns
        df_cnv_vars = df_cnv[meta_cols].copy() if meta_cols else pd.DataFrame(index=df_cnv.index)
        
        # Add one column per variant, copying from the mapped amplicon
        for _, row in map_df.iterrows():
            var = row["variant"]
            amp = row["amplicon"]
            
            if amp in df_cnv.columns:
                df_cnv_vars[var] = df_cnv[amp]
            else:
                # Amplicon missing -> fill with NaN
                df_cnv_vars[var] = np.nan
        
        vprint(f"    ✓ Created variant CNV: {df_cnv_vars.shape[0]} cells × {df_cnv_vars.shape[1]-len(meta_cols)} variants")
    
    # =========================================================================
    # Summary
    # =========================================================================
    vprint("\n" + "="*60)
    vprint("EXTRACTION COMPLETE")
    vprint("="*60)
    vprint(f"\n  df_cnv: {df_cnv.shape[0]} cells × {df_cnv.shape[1]-1} amplicons")
    vprint(f"  df_cnv_grouped: {df_cnv_grouped.shape[0]} amplicons × {df_cnv_grouped.shape[1]} clones")
    if df_cnv_vars is not None:
        n_meta = len([c for c in df_cnv_vars.columns if c in meta_cols])
        vprint(f"  df_cnv_vars: {df_cnv_vars.shape[0]} cells × {df_cnv_vars.shape[1]-n_meta} variants")
    else:
        vprint(f"  df_cnv_vars: Not created (no variant mapping provided)")
    vprint(f"  cnv_info: {len(cnv_info_sorted)} amplicons with genomic annotations")
    
    vprint("\n" + "="*60 + "\n")
    
    return df_cnv, df_cnv_grouped, df_cnv_vars, cnv_info_sorted


def _sync_dna_cnv_by_barcode(sample_obj, amplicons=None):
    """
    Keep only cells present in both DNA and CNV assays.

    Preserves DNA barcode order.
    Optionally subsets CNV columns to supplied amplicons.
    """
    dna_barcodes = np.asarray(sample_obj.dna.row_attrs["barcode"], dtype=str)
    cnv_barcodes = set(np.asarray(sample_obj.cnv.row_attrs["barcode"], dtype=str))

    shared_barcodes = [bc for bc in dna_barcodes if bc in cnv_barcodes]

    sample_obj.dna = sample_obj.dna[shared_barcodes, :]

    if amplicons is None:
        sample_obj.cnv = sample_obj.cnv[shared_barcodes, :]
    else:
        sample_obj.cnv = sample_obj.cnv[shared_barcodes, amplicons]

    return shared_barcodes


def _transfer_labels_by_barcode(source_assay, target_assay, label_col="label"):
    """Transfer row labels from source assay to target assay using barcode matching."""
    source_labels = pd.Series(
        source_assay.row_attrs[label_col],
        index=np.asarray(source_assay.row_attrs["barcode"], dtype=str),
    )
    target_barcodes = np.asarray(target_assay.row_attrs["barcode"], dtype=str)
    target_assay.row_attrs[label_col] = (
        pd.Series(target_barcodes)
        .map(source_labels)
        .to_numpy()
    )


def _get_labeled_attribute(assay, attr):
    """Return an assay attribute matrix with labels inserted as first column."""
    df = assay.get_attribute(attr, constraint="row+col").copy()
    cols = df.columns.tolist()
    df.insert(0, "label", assay.get_labels())
    return df[["label"] + cols]

def extract_cnv_data(
    Sample,
    desired_plotting_order=None,
    wd=None,
    EXPERIMENT=None,
    PATIENT_ID=None,
    OUT_DIR=None,
    output_tag="",
    sample_filter_source=None,
    label_source_sample=None,
    label_col="label",
    compute_ploidy=True,
    ploidy_before_sync=True,
    diploid_label="WT",
    normalized_counts="Normalized_reads",
    subset_cnv_to_variant_amplicons=True,
    save_outputs=True,
    verbose=True,
):
    """
    End-to-end CNV/ploidy extraction wrapper.

    Supports both workflows:
      1) Standard sample extraction.
      2) All-variant sample extraction with barcode restriction and label transfer
         from a filtered reference sample.

    Parameters
    ----------
    Sample : object
        Sample-like object with `.dna` and `.cnv` assays.
    desired_plotting_order : list
        Clone order passed to `get_cnv_info`.
    wd, EXPERIMENT, PATIENT_ID : str, optional
        Used to build OUT_DIR if OUT_DIR is not provided.
    OUT_DIR : str, optional
        Output directory. If None, built from wd/EXPERIMENT/PATIENT_ID.
    output_tag : str, default ""
        Suffix for output filenames (e.g., "_all_vars").
    sample_filter_source : object, optional
        If provided, restricts `Sample.dna` to barcodes present in
        `sample_filter_source.dna` (for all-variant workflows).
    label_source_sample : object, optional
        If provided, transfers labels by barcode from this sample into
        `Sample.dna` and `Sample.cnv`.
    label_col : str, default "label"
        Label column name in row_attrs.
    compute_ploidy : bool, default True
        Whether to compute ploidy using diploid reference cells.
    ploidy_before_sync : bool, default True
        If True, compute ploidy before DNA/CNV syncing (standard workflow).
        If False, sync first, compute ploidy, then re-sync (all-variant workflow).
    diploid_label : str or list of str, default "WT"
        Label(s) in `label_col` identifying the diploid reference population
        used for ploidy normalisation. A single string or a list of strings.
    normalized_counts : str, default "Normalized_reads"
        CNV layer name used for ploidy computation.
    subset_cnv_to_variant_amplicons : bool, default True
        Restrict CNV to DNA variant amplicons during syncing.
    save_outputs : bool, default True
        Save extracted tables to CSV.
    verbose : bool, default True
        Print progress logs.

    Returns
    -------
    dict
        Dictionary containing output paths and extracted dataframes, including:
        OUT_DIR, shared_barcodes, variant_amplicons, wt_cells,
        df_ploidy, df_ploidy_median,
        df_mut_to_amplicon_dataframe, df_mut_to_amplicon,
        variant_to_amp_dict,
        df_cnv, df_cnv_grouped, df_cnv_vars, cnv_info.
    """

    def vprint(msg):
        if verbose:
            print(msg)
            import sys
            sys.stdout.flush()

    if OUT_DIR is None:
        if wd is None or EXPERIMENT is None or PATIENT_ID is None:
            raise ValueError(
                "Provide OUT_DIR or all of wd, EXPERIMENT, and PATIENT_ID."
            )
        OUT_DIR = f"{wd}/results/per_sample/{EXPERIMENT}/{PATIENT_ID}/dna_cnv/clonotypes/ploidy"
    
    if desired_plotting_order is None:
        desired_plotting_order = sorted(set(Sample.dna.row_attrs['Clone']))

    os.makedirs(OUT_DIR, exist_ok=True)

    vprint("\n" + "=" * 60)
    vprint("EXTRACT_CNV_DATA")
    vprint("=" * 60)
    vprint(f"Output dir: {OUT_DIR}")

    # Optional pre-filter: restrict sample to barcodes present in reference sample
    if sample_filter_source is not None:
        source_barcodes = set(np.asarray(sample_filter_source.dna.row_attrs["barcode"], dtype=str))
        dna_barcodes = np.asarray(Sample.dna.row_attrs["barcode"], dtype=str)
        keep_barcodes = [bc for bc in dna_barcodes if bc in source_barcodes]
        Sample.dna = Sample.dna[keep_barcodes, :]
        vprint(f"Cells retained from sample_filter_source: {len(keep_barcodes)}")

    wt_cells = np.array([], dtype=object)

    # Workflow A: compute ploidy first, then sync (standard)
    if compute_ploidy and ploidy_before_sync:
        if isinstance(diploid_label, str):
            is_diploid = np.asarray(Sample.dna.row_attrs[label_col]) == diploid_label
        else:
            is_diploid = np.isin(np.asarray(Sample.dna.row_attrs[label_col]), diploid_label)

        wt_cells = Sample.dna.barcodes()[is_diploid]
        if len(wt_cells) == 0:
            raise ValueError(
                f"No cells with label(s) {diploid_label!r} found in '{label_col}'. "
                "Cannot compute ploidy without diploid reference cells."
            )
        vprint(f"Diploid reference cells ({diploid_label}): {len(wt_cells)}")
        Sample.cnv.compute_ploidy(
            diploid_cells=wt_cells,
            normalized_counts=normalized_counts,
        )

    variant_amplicons = None
    if subset_cnv_to_variant_amplicons:
        variant_amplicons = list(pd.unique(Sample.dna.col_attrs["amplicon"]))

    shared_barcodes = _sync_dna_cnv_by_barcode(Sample, amplicons=variant_amplicons)
    vprint(f"Shared DNA/CNV cells: {len(shared_barcodes)}")
    if variant_amplicons is not None:
        vprint(f"CNV amplicons retained: {len(variant_amplicons)}")

    # Label alignment
    if label_source_sample is not None:
        _transfer_labels_by_barcode(label_source_sample.dna, Sample.dna, label_col=label_col)
        _transfer_labels_by_barcode(label_source_sample.cnv, Sample.cnv, label_col=label_col)
        vprint(
            "DNA labels: "
            + str(pd.Series(Sample.dna.row_attrs[label_col]).value_counts(dropna=False).to_dict())
        )
        vprint(
            "CNV labels: "
            + str(pd.Series(Sample.cnv.row_attrs[label_col]).value_counts(dropna=False).to_dict())
        )
    else:
        _transfer_labels_by_barcode(Sample.dna, Sample.cnv, label_col=label_col)

    # Workflow B: sync first, compute ploidy, then re-sync (all-variant)
    if compute_ploidy and not ploidy_before_sync:
        if isinstance(diploid_label, str):
            is_diploid = np.asarray(Sample.dna.row_attrs[label_col]) == diploid_label
        else:
            is_diploid = np.isin(np.asarray(Sample.dna.row_attrs[label_col]), diploid_label)

        wt_cells = Sample.dna.barcodes()[is_diploid]
        if len(wt_cells) == 0:
            raise ValueError(
                f"No cells with label(s) {diploid_label!r} found in '{label_col}'. "
                "Cannot compute ploidy without diploid reference cells."
            )
        vprint(f"Diploid reference cells ({diploid_label}): {len(wt_cells)}")
        Sample.cnv.compute_ploidy(
            diploid_cells=wt_cells,
            normalized_counts=normalized_counts,
        )

        shared_barcodes = _sync_dna_cnv_by_barcode(Sample, amplicons=variant_amplicons)
        vprint(f"Shared DNA/CNV cells after ploidy: {len(shared_barcodes)}")
        _transfer_labels_by_barcode(Sample.dna, Sample.cnv, label_col=label_col)

    # Extract ploidy attributes
    df_ploidy = _get_labeled_attribute(Sample.cnv, "ploidy")
    df_ploidy_median = (
        df_ploidy
        .groupby("label", as_index=True)
        .median(numeric_only=True)
    )

    # Build variant -> amplicon map
    df_mut_to_amplicon_dataframe = pd.DataFrame(
        {
            "variant": Sample.dna.col_attrs["id"],
            "amplicon": Sample.dna.col_attrs["amplicon"],
        }
    )
    df_mut_to_amplicon = df_mut_to_amplicon_dataframe.to_dict(orient="list")
    variant_to_amp_dict = dict(
        zip(df_mut_to_amplicon["variant"], df_mut_to_amplicon["amplicon"])
    )

    map_filename = f"Amplicons_to_variant{output_tag}.csv" if output_tag else "Amplicons_to_variant.csv"
    amplicon_map_path = f"{OUT_DIR}/{map_filename}"
    if save_outputs:
        df_mut_to_amplicon_dataframe.to_csv(amplicon_map_path, index=False)
        vprint(f"Saved: {amplicon_map_path}")

    # Extract CNV/ploidy summary tables
    df_cnv, df_cnv_grouped, df_cnv_vars, cnv_info = get_cnv_info(
        Sample=Sample,
        df_mut_to_amplicon=df_mut_to_amplicon,
        desired_plotting_order=desired_plotting_order,
        verbose=verbose,
    )

    # Save outputs
    outputs = {
        f"Cell_x_ploidy_data{output_tag}.csv": df_cnv,
        f"Clone_x_ploidy_data{output_tag}.csv": df_cnv_grouped,
        f"Variants_x_ploidy_data{output_tag}.csv": df_cnv_vars,
        f"Ploidy_metadata{output_tag}.csv": cnv_info,
    }

    if save_outputs:
        for filename, df_out in outputs.items():
            path = f"{OUT_DIR}/{filename}"
            df_out.to_csv(path)
            vprint(f"Saved: {path}")

    return {
        "OUT_DIR": OUT_DIR,
        "shared_barcodes": shared_barcodes,
        "variant_amplicons": variant_amplicons,
        "wt_cells": wt_cells,
        "df_ploidy": df_ploidy,
        "df_ploidy_median": df_ploidy_median,
        "df_mut_to_amplicon_dataframe": df_mut_to_amplicon_dataframe,
        "df_mut_to_amplicon": df_mut_to_amplicon,
        "variant_to_amp_dict": variant_to_amp_dict,
        "df_cnv": df_cnv,
        "df_cnv_grouped": df_cnv_grouped,
        "df_cnv_vars": df_cnv_vars,
        "cnv_info": cnv_info,
        "outputs": outputs,
    }
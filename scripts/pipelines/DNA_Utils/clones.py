"""Clone-level analyses: patient assignment, clone tables, VAF, and sizing."""

import numpy as np
import pandas as pd

from .utils import drop_scalar_row_attrs


def assign_patients_from_whitelist(
    dna_obj,
    samples_whitelist: dict,
    *,
    af_attr: str = "AF",
    label_key: str = "label",
    af_cutoff: float = 10,
) -> dict:
    """Relabel clusters with Patient_IDs based on a variant whitelist.

    Returns
    -------
    dict
        ``{old_cluster_label: new_patient_id | 'Unknown'}``
    """
    wl_sets = {
        pid: {v.strip() for v in str(vlist).split(",") if v.strip()}
        for pid, vlist in samples_whitelist.items()
    }

    af = dna_obj.get_attribute(af_attr, constraint="row+col")
    labels = dna_obj.row_attrs[label_key]

    cluster_hits = {}
    for cluster in np.unique(labels):
        rows = labels == cluster
        mean_af = af.loc[rows, :]
        wl_cols = [v for s in wl_sets.values() for v in s if v in mean_af.columns]
        mean_af_wl = mean_af.loc[:, wl_cols]
        high_vars = mean_af_wl.mean(axis=0)[lambda s: s > af_cutoff].index.tolist()
        cluster_hits[cluster] = high_vars

    cluster_to_patient = {}
    for cluster, hit_list in cluster_hits.items():
        hit_set = set(hit_list)
        best_pid, best_score = None, -1.0
        for pid, var_set in wl_sets.items():
            if not var_set:
                continue
            score = len(hit_set & var_set) / len(var_set)
            if score > best_score or (
                score == best_score
                and len(var_set) > len(wl_sets.get(best_pid, set()))
            ):
                best_pid, best_score = pid, score
        cluster_to_patient[cluster] = best_pid if best_score > 0 else "Unknown"

    dna_obj.rename_labels(cluster_to_patient)
    return cluster_to_patient


def clones_to_dataframe(wf, variants, max_ado_score: float = 0.8,
                        min_clone_size: int = 1) -> pd.DataFrame:
    """Call ``wf.get_clones(…)`` and return a tidy DataFrame with display
    column names and UI order preserved."""

    def _normalize(v):
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

    variants = _normalize(variants)
    seen, clean = set(), []
    for s in (x.strip() for x in variants if x is not None):
        if s and s not in seen:
            seen.add(s)
            clean.append(s)
    variants = clean

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
    clone_order: list = None,
    output_path: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute variant allele frequencies per clone.

    Returns
    -------
    df_vaf : pd.DataFrame
        Variants (rows) × clones (columns), values are VAF %.
    """
    import copy, os

    def vprint(msg):
        if verbose:
            print(msg)

    vprint("\n" + "=" * 60)
    vprint("COMPUTING VAF PER CLONE")
    vprint("=" * 60)

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

    vprint("=" * 60 + "\n")
    return df_vaf


def get_clones_size(
    Sample,
    label_col: str = "label",
    desired_order: list = None,
    exclude_celltypes: list = None,
    return_percentage: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Calculate absolute cell counts (and optional %) per clone.

    Parameters
    ----------
    exclude_celltypes : list, optional
        Protein labels to exclude before counting.

    Returns
    -------
    label_counts : pd.DataFrame
        Columns: ``Clone``, ``Count`` (and optionally ``Percentage``).
    """
    import copy

    def vprint(msg):
        if verbose:
            print(msg)

    vprint("\n" + "=" * 60)
    vprint("CALCULATING CLONE SIZE")
    vprint("=" * 60)

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
    vprint("=" * 60 + "\n")
    return label_counts

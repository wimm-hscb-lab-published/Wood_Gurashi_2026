import numpy as np
import pandas as pd


def update_protein_labels_from_dataframe(
    Sample,
    anno_df: pd.DataFrame,
    barcode_col: str = None,
    label_col: str = 'label',
    target_attr: str = 'label',
    verbose: bool = True
) -> None:
    """
    Update Sample.protein.row_attrs with labels from an annotation dataframe.
    Matches cells by barcode.

    Parameters
    ----------
    Sample : object
        Sample object with protein attribute
    anno_df : pd.DataFrame
        DataFrame containing barcodes and labels.
        If barcode_col is None, assumes barcodes are in the index.
    barcode_col : str, optional
        Name of column containing barcodes in anno_df.
        If None, uses the index.
    label_col : str
        Name of column containing labels in anno_df (default: 'label')
    target_attr : str
        Name of attribute to update in Sample.protein.row_attrs (default: 'label')
    verbose : bool
        Print progress messages

    Returns
    -------
    None
        Updates Sample.protein.row_attrs[target_attr] in place

    Example
    -------
    >>> update_protein_labels_from_dataframe(
    ...     Sample=Sample,
    ...     anno_df=pro_df_anno,
    ...     label_col='label'
    ... )
    """
    import sys

    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    vprint("\n" + "=" * 60)
    vprint("UPDATING PROTEIN LABELS FROM DATAFRAME")
    vprint("=" * 60)

    # Get barcodes from annotation dataframe
    if barcode_col is None:
        anno_barcodes = anno_df.index.values
        vprint("  Using index as barcodes")
    else:
        if barcode_col not in anno_df.columns:
            raise ValueError(f"Column '{barcode_col}' not found in anno_df")
        anno_barcodes = anno_df[barcode_col].values
        vprint(f"  Using column '{barcode_col}' as barcodes")

    # Get labels from annotation dataframe
    if label_col not in anno_df.columns:
        raise ValueError(f"Column '{label_col}' not found in anno_df")
    anno_labels = anno_df[label_col].values

    vprint(f"  Annotation dataframe: {anno_df.shape[0]} cells")
    vprint(f"  Unique labels in dataframe: {len(np.unique(anno_labels))}")

    # Get barcodes from Sample.protein
    protein_barcodes = np.array(Sample.protein.barcodes())
    vprint(f"\n  Sample.protein: {len(protein_barcodes)} cells")

    # Create mapping dictionary
    anno_dict = dict(zip(anno_barcodes, anno_labels))

    # Map labels to protein barcodes
    new_labels = []
    n_matched = 0
    n_missing = 0

    for bc in protein_barcodes:
        if bc in anno_dict:
            new_labels.append(anno_dict[bc])
            n_matched += 1
        else:
            if target_attr in Sample.protein.row_attrs:
                existing_label = Sample.protein.row_attrs[target_attr][
                    protein_barcodes == bc
                ][0]
                new_labels.append(existing_label)
            else:
                new_labels.append('Unknown')
            n_missing += 1

    new_labels = np.array(new_labels)

    vprint(f"\n  Matching results:")
    vprint(f"    Matched: {n_matched} cells ({100 * n_matched / len(protein_barcodes):.1f}%)")
    vprint(f"    Not matched: {n_missing} cells ({100 * n_missing / len(protein_barcodes):.1f}%)")

    # Update Sample.protein.row_attrs
    Sample.protein.add_row_attr(target_attr, new_labels)

    vprint(f"\n  ✓ Updated Sample.protein.row_attrs['{target_attr}']")

    # Print label distribution
    unique_labels, counts = np.unique(new_labels, return_counts=True)
    vprint(f"\n  Label distribution:")
    for label, count in sorted(zip(unique_labels, counts), key=lambda x: -x[1]):
        vprint(f"    {label:20s}: {count:5d} cells ({100 * count / len(new_labels):5.1f}%)")

    vprint("\n" + "=" * 60)
    vprint("✓ LABELS UPDATED")
    vprint("=" * 60)

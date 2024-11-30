import bioframe as bf
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests

from paired_interactions import (
    bin_contacts,
    find_close_pairs,
    get_peaks_coords_wo_shift,
    nonzero_vec_to_sparse,
)


def calc_expected(
    annotation_df: pd.DataFrame,
    fithic_df: pd.DataFrame,
    bins_df: pd.DataFrame,
    chromosome: str,
    bin_size: int = 10000,
    diag_threshold: int = 2,
) -> dict:
    """Calculates expected probabilities of states pairs based on Hi-C peaks number.

    Args:
        annotation_df (pd.DataFrame): DF with state names, chrs, starts and ends.
        fithic_df (pd.DataFrame): DF with Fit-Hi-C peaks.
        bins_df (pd.DataFrame): DF with bins chrs, starts and ends.
        chromosome (str): Selected chromosome name (1,2...X).
        bin_size (int, optional): Number of b.p. per bin. Defaults to 10000.
        diag_threshold (int, optional): How many diagonals from the main to remove. Defaults to 2.

    Returns:
        dict: States names with Hi-C peaks proportions which interect with states pairs.
    """
    bins_chr = bins_df[bins_df.chrom == f"chr{chromosome}"]
    bins_number = len(bins_chr)
    hic_coords = get_peaks_coords_wo_shift(fithic_df, chromosome, bin_size)
    hic_matrix = np.full((bins_number, bins_number), False)
    hic_matrix[hic_coords] = True
    hic_matrix = np.tril(hic_matrix, k=-diag_threshold)
    close_pairs_num = hic_matrix.sum()
    annotation_chr = annotation_df[annotation_df.chr == f"chr{chromosome}"]

    states = list(annotation_df.state.unique())
    states_exp = {state: 0 for state in states}
    for state in annotation_chr.state.unique():
        state_coords = annotation_chr.loc[
            annotation_chr.state == state, ["chr", "start", "end"]
        ]
        state_bins = bf.overlap(
            bins_chr,
            state_coords,
            how="inner",
            cols1=["chrom", "start", "end"],
            cols2=["chr", "start", "end"],
            suffixes=["", "_"],
        )[["start", "end"]].drop_duplicates()
        state_bins = np.array(
            ((state_bins.start + state_bins.end) / 2) // bin_size, dtype="int32"
        )
        state1 = np.repeat(state_bins, len(state_bins))
        state2 = np.tile(state_bins, len(state_bins))
        close_state_pairs_num = hic_matrix[(state1, state2)].sum()
        state_exp = close_state_pairs_num / close_pairs_num
        states_exp[state] = state_exp

    return states_exp


def calc_paired_contacts_in_states(
    contacts_df: pd.DataFrame,
    rnas_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    fithic_df: pd.DataFrame,
    bins_df: pd.DataFrame,
    bad_bins: np.ndarray,
    chromosome: str,
    bin_size: int = 10000,
    diag_threshold: int = 2,
) -> pd.DataFrame:
    """Calculates number of paired RD-contacts on a chromosome, which intersect with each state of annotation.

    Args:
        contacts_df (pd.DataFrame): DF of contacts position on a single chromosome.
        rnas_df (pd.DataFrame): DF with RNAs associated with chromatin structure, consists of RNA name chromosome of interactions
        and paired interactions numbers.
        annotation_df (pd.DataFrame): DF with state names, chrs, starts and ends.
        fithic_df (pd.DataFrame): DF with Fit-Hi-C peaks.
        bins_df (pd.DataFrame):  DF with bins chrs, starts and ends.
        bad_bins (np.ndarray): Bins nto filter out (for example with low coverage).
        chromosome (str): Selected chromosome name (1,2...X).
        bin_size (int, optional): Number of b.p. per bin. Defaults to 10000.
        diag_threshold (int, optional): How many diagonals from the main to remove. Defaults to 2.

    Returns:
        pd.DataFrame: DF with number of paired interactions for each state
    """
    states = list(annotation_df.state.unique())
    bins_chr = bins_df[bins_df.chrom == f"chr{chromosome}"]
    bins_number = len(bins_chr)
    hic_coords = get_peaks_coords_wo_shift(fithic_df, chromosome, bin_size)
    hic_matrix = np.full((bins_number, bins_number), False)
    hic_matrix[hic_coords] = True
    hic_matrix = np.tril(hic_matrix, k=-diag_threshold)
    rnas_df_chr = rnas_df[rnas_df.chr_of_contacts == str(chromosome)]
    rna_list = rnas_df_chr.gene_name_un.to_list()
    results = rnas_df_chr[["gene_name_un", "chr_of_contacts", "close_pairs_num"]]
    annotation_chr = annotation_df[annotation_df.chr == f"chr{chromosome}"]
    for state in states:
        results[state] = 0
        state_coords = annotation_chr.loc[
            annotation_chr.state == state, ["chr", "start", "end"]
        ]
        # get genome bins intersecting with current annotation state
        state_bins = bf.overlap(
            bins_chr,
            state_coords,
            how="inner",
            cols1=["chrom", "start", "end"],
            cols2=["chr", "start", "end"],
            suffixes=["", "_"],
        )[["start", "end"]].drop_duplicates()
        state_bins = np.array(
            ((state_bins.start + state_bins.end) / 2) // bin_size, dtype="int32"
        )

        # creating states pairs
        state1 = np.repeat(state_bins, len(state_bins))
        state2 = np.tile(state_bins, len(state_bins))
        state_mask = np.full(shape=hic_matrix.shape, fill_value=False)
        state_mask[(state1, state2)] = True
        close_state_coords = (hic_matrix & state_mask).nonzero()

        # counting observed RD-paired contacts in states
        for rna in rna_list:
            contacts_counts, contacts_positions = bin_contacts(
                contacts_df, rna, bad_bins, bin_size
            )
            rna_contacts_sparse = nonzero_vec_to_sparse(
                contacts_counts,
                contacts_positions,
                bins_number,
                diag_threshold,
            )
            close_pairs, _ = find_close_pairs(rna_contacts_sparse, close_state_coords)
            results.loc[results.gene_name_un == rna, state] = close_pairs
    return results


def chi_square_enrichment(
    results: pd.DataFrame, states: list, probs: bool = True
) -> pd.DataFrame:
    """Computes chi-square pvalues using expected probabilities.


    Args:
        results (pd.DataFrame): DF with observed pair interactions for RNA per state merged with expected probabilities
        for states on a chr based on Hi-C interactions.
        states (list): List of states for the annotation.
        probs(bool): If expected values are probabilites or interactions.

    Returns:
        pd.DataFrame: DF with added p-values for each state.
    """
    results_chi2 = results.copy()
    for state in states:
        results_chi2[f"{state}_pvalue"] = np.nan
        if probs:
            results_chi2[f"{state}_exp"] *= results_chi2.close_pairs_num + len(
                states
            )  # + len(states) to count pseudocounts
        for i in range(results_chi2.shape[0]):
            state_o = results_chi2.loc[i, state]
            state_e = results_chi2.loc[i, f"{state}_exp"]
            other_o = (
                results_chi2.loc[i, "close_pairs_num"] - results_chi2.loc[i, state]
            )
            other_e = (
                results_chi2.loc[i, "close_pairs_num"]
                - results_chi2.loc[i, f"{state}_exp"]
            )
            stat, pvalue = chisquare(f_obs=[state_o, other_o], f_exp=[state_e, other_e])
            results_chi2.loc[i, f"{state}_pvalue"] = pvalue

    return results_chi2


def correct_pvalues(
    results_chi: pd.DataFrame, states: list, return_oe: bool = False
) -> pd.DataFrame:
    """Performs Benjamini-Hocheberg correction with all pvalues for annotation.


    Args:
        results_chi (pd.DataFrame): DF with chi-square test p-values.
        states (list): List of states for the annotation.
        return_oe (bool): Return o/e values or not.

    Returns:
        pd.DataFrame: Corrected p-values in long format.
    """
    results_long = []
    for state in states:
        state_res = results_chi[["gene_name_un", "chr_of_contacts", "close_pairs_num"]]
        state_res["state"] = state
        state_res["pvalue"] = results_chi[f"{state}_pvalue"]
        if return_oe:
            state_res["oe"] = results_chi[f"{state}_oe"]
        results_long.append(state_res)

    results_long = pd.concat(results_long)
    results_long_corrected = results_long.copy()
    results_long_corrected.loc[:, "pvalue"] = multipletests(
        pvals=results_long_corrected["pvalue"],
        alpha=0.05,
        method="fdr_bh",
        is_sorted=False,
    )[1]

    return results_long_corrected


def construct_heatmap(
    results: pd.DataFrame, expected: pd.DataFrame, states: list, filter: bool = True
) -> pd.DataFrame:
    """Constructs table for visualization for selected annotation, insignificant results = 0.

    Args:
        results (pd.DataFrame): DF with observed pair interactions for RNA per state.
        expected (pd.DataFrame): Expected probabilities based on Hi-C interactions.
        states (list): List of states for the annotation.
        filter (bool): Filter insignificant cells.

    Returns:
        pd.DataFrame: DF with O/E for RNAs for state, with zeros for insignificant values.
    """

    PSEUDOCOUNT = 1e-6
    results[states] += 1  # for log2
    expected = expected[[*states, "chr"]]
    expected = expected.rename({state: f"{state}_exp" for state in states}, axis=1)
    for state in states:
        expected.loc[expected[f"{state}_exp"] == 0, f"{state}_exp"] = PSEUDOCOUNT
    results["chr_of_contacts"] = results["chr_of_contacts"].astype("string")
    expected["chr"] = expected["chr"].astype("string")
    results_merged = results.merge(
        expected,
        how="left",
        left_on="chr_of_contacts",
        right_on="chr",
        suffixes=("", "_"),
    ).drop("chr_", axis=1)

    # chi-square test computation
    results_chi2 = chi_square_enrichment(results_merged, states)

    # correcting p-vals
    corrected_chi2 = correct_pvalues(results_chi2, states)

    # filtering insignificant
    for state in states:
        results_merged[f"{state}_exp"] *= results_merged.close_pairs_num + len(
            states
        )  # + len(states) to count pseudocounts

        results_merged[f"{state}_oe"] = np.log2(
            results_merged[state] / results_merged[f"{state}_exp"]
        )
        significance = corrected_chi2.loc[
            (corrected_chi2["state"] == state),
            ["gene_name_un", "chr_of_contacts", "pvalue"],
        ]
        results_merged = results_merged.merge(
            significance, how="left", on=["gene_name_un", "chr_of_contacts"]
        )
        if filter:
            results_merged.loc[results_merged["pvalue"] >= 0.05, f"{state}_oe"] = 0
        results_merged = results_merged.drop(["pvalue"], axis=1)

    results_merged = results_merged[
        [
            "gene_name_un",
            "chr_of_contacts",
            "close_pairs_num",
            *list([f"{state}_oe" for state in states]),
        ]
    ]

    return results_merged

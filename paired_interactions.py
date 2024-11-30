import numpy as np
import pandas as pd
from scipy import sparse


def bin_contacts(
    contacts_df: pd.DataFrame,
    rna_name: str,
    bad_bins: np.ndarray,
    bin_size: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Function to bin track of RNA contucts.

    Args:
        contacts (pd.DataFrame): DF of contacts position on a single chromosome.
        rna_name (str): Name of selected RNA gene.
        bad_bins (np.ndarray): Bins nto filter out (for example with low coverage).
        bin_size (int, optional): Size of bin. Defaults to 10000.

    Returns:
        tuple[np.ndarray, np.ndarray]: Array of bins contacts counts and array of bins positions.
    """
    contacts_df["center"] = (contacts_df.dna_start + contacts_df.dna_end) / 2
    contacts_df["bin"] = (contacts_df.center // bin_size).astype(int)
    rna_contacts = contacts_df.loc[contacts_df.gene_name_un == rna_name, ["bin"]]
    rna_contacts = rna_contacts.groupby("bin").value_counts()
    rna_contacts = rna_contacts[~np.isin(rna_contacts.index, bad_bins)]

    return np.array(rna_contacts.values), np.array(rna_contacts.index)


def get_peaks_coords_wo_shift(
    fithic_df: pd.DataFrame,
    chromosome: str,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns real coordinates of close bins.


    Args:
        fithic_df (pd.DataFrame): Dataframe of Fit-Hi-C peaks coordinates.
        chromosome (str): Chromosome (only number)).
        resolution (int): Number of b.p. per bin.

    Returns:
       tuple[np.ndarray, np.ndarray]: Real Fit-Hi-C peaks coordinates.
    """

    fithic_df = fithic_df[fithic_df.chr1 == str(chromosome)]
    hic_coords = (
        np.array(fithic_df.fragmentMid2) // resolution,
        np.array(fithic_df.fragmentMid1) // resolution,
    )

    return hic_coords


def get_peaks_coords(
    fithic_df: pd.DataFrame,
    compartments: pd.DataFrame,
    chromosome: str,
    shift: int,
    bins_number: int,
    resolution: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Returns real and shifted coordinates of close bins.


    Args:
        fithic_df (pd.DataFrame): Dataframe of Fit-Hi-C peaks coordinates.
        compartments (pd.DataFrame): Dataframe of coordinates of A/B compartments.
        chromosome (str): Chromosome (only number)).
        shift (int): Number of bins to shift Hi-C peaks.
        bins_number (int): Number of bins on the chromosome.
        resolution (int): Number of b.p. per bin.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]: Real and shifted Fit-Hi-C peaks coordinates.
    """
    A_coords = compartments.loc[(compartments["Chr"] == str(chromosome)), "Vec"] == 1
    B_coords = compartments.loc[(compartments["Chr"] == str(chromosome)), "Vec"] == 0
    fithic_df = fithic_df[fithic_df.chr1 == str(chromosome)]
    hic_coords = (
        np.array(fithic_df.fragmentMid2) // resolution,
        np.array(fithic_df.fragmentMid1) // resolution,
    )
    coords = np.arange(bins_number)
    coords_rolled = np.array(coords, copy=True)
    coords_rolled[A_coords] = np.roll(coords_rolled[A_coords], shift)
    coords_rolled[B_coords] = np.roll(coords_rolled[B_coords], shift)
    coords_df = pd.DataFrame({"x": hic_coords[0], "y": hic_coords[1]})
    df_shift_x = pd.DataFrame({"coords": coords_rolled, "new_x": coords})
    df_shift_y = pd.DataFrame({"coords": coords_rolled, "new_y": coords})
    coords_df = coords_df.merge(df_shift_x, how="left", left_on="x", right_on="coords")
    coords_df = coords_df.merge(df_shift_y, how="left", left_on="y", right_on="coords")
    hic_shifted_coords = (np.array(coords_df.new_x), np.array(coords_df.new_y))

    return hic_coords, hic_shifted_coords


def nonzero_vec_to_sparse(
    rnadna_vec: np.ndarray, rnadna_coords: np.ndarray, length: int, diag_threshold: int
) -> sparse.coo_matrix:
    """Convert binned vector of contacts to matrix of pairs.

    Args:
        rnadna_vec (np.ndarray): Contacts counts per bin.
        rnadna_coords (np.ndarray): Coordinates of bins.
        length (int): Length of chromosome in bins.
        diag_threshold (int): How many diagonals from the main to remove.

    Returns:
        sparse.coo_matrix: Sparse matrix of pairs.
    """
    I = np.repeat(rnadna_coords, len(rnadna_coords))
    J = np.tile(rnadna_coords, len(rnadna_coords))
    V = np.outer(rnadna_vec, rnadna_vec).flatten("C")
    rnadna_sparse = sparse.coo_matrix((V, (I, J)), shape=(length, length)).tocsr()
    rnadna_sparse = sparse.tril(rnadna_sparse, k=-diag_threshold, format="csr")
    return rnadna_sparse


def find_close_pairs(
    rnadna_sparse: sparse.csr_matrix, hic_coords: tuple[int, int]
) -> tuple[int, int]:
    """Returns number of close and far pairs of RNA-DNA contacts based on coordinates of Hi-C map.

    Args:
        rnadna_sparse (sparse.coo_matrix): Sparse matrix of pairs.
        hic_coords (tuple[np.ndarray, np.ndarray]): Tuple of arrays of Hi-C peaks coordinates.

    Returns:
        tuple[int, int]: Close pairs number and far pairs number.
    """
    rnadna_sum = rnadna_sparse.sum()
    close_pairs = rnadna_sparse[hic_coords].sum()
    far_pairs = rnadna_sum - close_pairs
    return close_pairs, far_pairs


# Functions for paired contacts from different chromosomes


def get_trans_peaks_coords(
    fithic_df: pd.DataFrame,
    compartments: pd.DataFrame,
    chromosome1: str,
    chromosome2: str,
    shift: int,
    bins_number1: int,
    bins_number2: int,
    resolution: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Returns real and shifted coordinates of close bins of trans Hi-C.

    Args:
        fithic_df (pd.DataFrame): Dataframe of Fit-Hi-C peaks coordinates.
        compartments (pd.DataFrame): Dataframe of coordinates of A/B compartments.
        chromosome1 (str): 1st chromosome (only number)).
        chromosome2 (str): 2nd chromosome (only number)).
        shift (int): Number of bins to shift Hi-C peaks.
        bins_number1 (int): Number of bins on the 1st chromosome.
        bins_number2 (int): Number of bins on the 2nd chromosome.
        resolution (int): Number of b.p. per bin.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]: _description_
    """
    fithic_df_chr = fithic_df[
        (fithic_df.chr1 == str(chromosome1)) & (fithic_df.chr2 == str(chromosome2))
    ]
    hic_coords = (
        np.array(fithic_df_chr.fragmentMid1) // resolution,
        np.array(fithic_df_chr.fragmentMid2) // resolution,
    )
    A_coords1 = (
        compartments.loc[(compartments["Chr"] == f"chr{chromosome1}"), "Vec"] == 1
    )
    B_coords1 = (
        compartments.loc[(compartments["Chr"] == f"chr{chromosome1}"), "Vec"] == 0
    )
    A_coords2 = (
        compartments.loc[(compartments["Chr"] == f"chr{chromosome2}"), "Vec"] == 1
    )
    B_coords2 = (
        compartments.loc[(compartments["Chr"] == f"chr{chromosome2}"), "Vec"] == 0
    )
    coords1 = np.arange(bins_number1)
    coords2 = np.arange(bins_number2)
    coords_rolled1 = np.array(coords1, copy=True)
    coords_rolled2 = np.array(coords2, copy=True)
    coords_rolled1[A_coords1] = np.roll(coords_rolled1[A_coords1], shift)
    coords_rolled1[B_coords1] = np.roll(coords_rolled1[B_coords1], shift)
    coords_rolled2[A_coords2] = np.roll(coords_rolled2[A_coords2], shift)
    coords_rolled2[B_coords2] = np.roll(coords_rolled2[B_coords2], shift)
    coords_df = pd.DataFrame({"x": hic_coords[0], "y": hic_coords[1]})
    df_shift_x = pd.DataFrame({"coords": coords_rolled1, "new_x": coords1})
    df_shift_y = pd.DataFrame({"coords": coords_rolled2, "new_y": coords2})
    coords_df = coords_df.merge(df_shift_x, how="left", left_on="x", right_on="coords")
    coords_df = coords_df.merge(df_shift_y, how="left", left_on="y", right_on="coords")
    hic_shifted_coords = (np.array(coords_df.new_x), np.array(coords_df.new_y))

    return hic_coords, hic_shifted_coords


def trans_vec_to_sparse(
    rnadna_vec1: np.ndarray,
    rnadna_coords1: np.ndarray,
    length1: int,
    rnadna_vec2: np.ndarray,
    rnadna_coords2: np.ndarray,
    length2: int,
) -> sparse.csr_array:
    """Convert binned vector of contacts to matrix of pairs.


    Args:
        rnadna_vec1 (np.ndarray): Contacts counts per bin on 1st chromosome.
        rnadna_coords1 (np.ndarray): Coordinates of bins on 1st chromosome.
        length1 (int):  Length of 1st chromosome in bins.
        rnadna_vec2 (np.ndarray): Contacts counts per bin on 2nd chromosome.
        rnadna_coords2 (np.ndarray): Coordinates of bins on 2nd chromosome.
        length2 (int): Length of 2nd chromosome in bins.

    Returns:
        sparse.csr_array: Sparse matrix of pairs.
    """
    I = np.repeat(rnadna_coords1, len(rnadna_coords2))
    J = np.tile(rnadna_coords2, len(rnadna_coords1))
    V = np.outer(rnadna_vec1, rnadna_vec2).flatten("C")
    rnadna_sparse = sparse.coo_matrix(
        (V, (I, J)), shape=(length1, length2), dtype="float64"
    ).tocsr()
    return rnadna_sparse

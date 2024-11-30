# RNA_DNA_HiC
This is a repo for functions for paired RNA-DNA interactions analysis used in "Joint analysis of RNA-DNA and DNA-DNA interactomes reveals their strong association" article.

## paired_interactions.py

- `bin_contacts` function to bin track of RNA contucts with single bin size.
- `get_peaks_coords_wo_shift` function to get arrays of coordinated of Fit-Hi-C significant interactions.
- `get_peaks_coords` function to get arrays of real and shifted coordinated of Fit-Hi-C significant interactions.
- `nonzero_vec_to_sparse` function to convert binned vector of RNA-DNA contacts to matrix of RD-pairs.
- `find_close_pairs` function to calculate number of close and far pairs of RNA-DNA contacts based on coordinates of Fit-Hi-C significant interactions.
- `get_trans_peaks_coords` function to get arrays of real and shifted coordinated of trans Fit-Hi-C significant interactions.
- `trans_vec_to_sparse` function to convert two binned vectors of RNA-DNA contacts from different chromosomes to matrix of RD-pairs.

## annotations.py
- `calc_expected` function to calculate expected probabilities of states pairs based on Fit-Hi-C significant interactions which intersect with states pairs.
- `calc_paired_contacts_in_states` fucntion to calculates number of paired RD-contacts on a chromosome, which intersect with each state of annotation.
- `chi_square_enrichment` function to compute chi-square pvalues for each RNA using expected probabilities of state pairs.
- `correct_pvalues` function to performs Benjamini-Hocheberg correction with all pvalues for annotation.
- `construct_heatmap` function to construct table for visualization of observed over expected values for each RNA for each state in selected annotation.

# Examples of data formats

## Fit-Hi-C significant interactions

|chr1|fragmentMid1|chr2|fragmentMid2|q-value|
|----|-----|---|----|-----|
|1|825000|1|825000|1.1883280000000002e-69|
|1|825000|1|835000|2.81959e-14|

## More examples coming soon

## Overview
This folder contains processed datasets and the workflow for the analysis conducted in Section 6 of the paper. Each folder corresponds to a dataset. Additional details for each dataset can be found in the `Readme.md` files within.

### MNIST
The script `MNIST_analysis.m` fits the dataset via the proposed method as well as produces all results (excluding comparison with other methods) in the main text as well as the supplement. This includes Tables 2, 4 and Figures 3, 5, S.16. Here, the `MNIST_analysis_data.mat` data file stores the estimated coefficients and posterior probabilities.

### 20 newsgroups
The script TBA..

### TIMSS
The script `TIMSS_analysis.m` computes the average latent variable estimates per question in Table 6, as well as generates the heatmaps in Figure S.16.

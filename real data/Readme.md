## Overview
This folder contains processed datasets and the workflow for the analysis conducted in Section 6 of the paper. Each folder corresponds to a dataset. Additional details for each dataset can be found in the `Readme.md` files within. The code for implementing other methods is not provided here; however, the reproducibility details are outlined in the written supplement.

### MNIST
The script `MNIST_analysis.m` fits the dataset via the proposed method, as well as produces all results in the main text as well as the supplement. This includes Tables 2, 4 and Figures 3, 5, S.16. Here, the `MNIST_analysis_data.mat` data file stores the estimated coefficients and posterior probabilities.

### 20 newsgroups
The script `20_newsgroups_analysis.m` fits the dataset via the proposed method as well as produces all results (Figures 1, 6 and Table 5) in the main text. Here, the `MNIST_analysis_data.mat` data file stores the estimated coefficients and posterior probabilities. Here, the `20_newsgroups_data.mat` data file stores the estimated coefficients and posterior probabilities.

### TIMSS
The script `TIMSS_analysis.m` computes the average latent variable estimates per question in Table 6, as well as generates the heatmaps in Figure S.16.

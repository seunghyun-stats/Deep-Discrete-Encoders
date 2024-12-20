# Deep-Discrete-Encoders

This repository contains the MATLAB codes for the manuscript "Deep Discrete Encoders: Identifiable Deep Generative
Models for Rich Data with Discrete Latent Layers"

### Codes for the proposed method:
The folder `main algorithms` contains the codes for Algorithms 1-3 (basic PEM, double-SVD initialization, penalized SAEM) proposed in the manuscript. Each sub-folder (Bernoulli, Poisson, Normal) corresponds to each observed-layer parametric family.

### For simulations:
To run simulations, go to the folder `simulations` and run the script corresponding to the response type of interest (Bernoulli, Poisson, Normal).

### For real data analysis:
To replicate data analysis, go to the folder `real data` and select the dataset of interest (MNIST, 20 newsgroups, TIMSS). Alongside the main script, the processed dataset and supplementary functions are also provided.

### Additional codes:
The folder `utilities` contains sub-functions that will be required to implement the main algorithms. We recommend adding this folder using the `addpath 'utilities'` command in MATLAB.

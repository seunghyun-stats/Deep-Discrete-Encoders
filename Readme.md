# Deep-Discrete-Encoders

This repository contains the MATLAB codes for the manuscript [Lee, S. and Gu, Y. (2025), Deep Discrete Encoders: Identifiable Deep Generative Models for Rich Data with Discrete Latent Layers](https://arxiv.org/abs/2501.01414).

### Codes for the proposed method:
The folder `main algorithms` contains the codes for the proposed computation pipeline (Algorithms 1-2 in the manuscript). Each sub-folder (Bernoulli, Poisson, Normal) corresponds to each observed-layer parametric family.

### For simulations:
To run simulations, go to the folder `simulations` and run the script corresponding to each simulation setting. Additional details are provided in the `Readme.md` file inside the folder.

### For real data analysis:
To replicate data analysis, go to the folder `real data` and select the dataset of interest (MNIST, 20 newsgroups, TIMSS). Alongside the main script, the processed dataset and supplementary functions are also provided. Additional details are provided in the `Readme.md` file inside the folder.

### Additional codes:
The folder `utilities` contains sub-functions that will be required to implement the main algorithms. Before running the scripts for simulations/real data analysis, we recommend adding this folder using the `addpath 'utilities'` command in MATLAB.

### Dependencies:
MATLAB  9.13.0.2126072 (R2022b)

Optimization toolbox 9.4,
Parallel Computing Toolbox 7.7,
Statistics and Machine Learning Toolbox 12.4

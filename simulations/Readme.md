## Overview
The `simulations` folder contains two sub-folders, which contain codes used for the Simulation Studies section (Section 5) in the main paper as well as related supplements.

### 1. general response types
This folder contains three scripts for estimating two-latent-layer DDEs, which corresponds to the first paragraph in Section 5.
Each file (Bernoulli, Normal, Poisson) corresponds to the simulation results presented in Figure 4 in the main paper (and the exact values displayed in Tables S.6-11).
To reproduce the results, run the script corresponding to the observed-layer parametric family of interest (e.g. Bernoulli).

### 2. deeper models
This folder contains two scripts for experiments of DDEs with more than two latent layers, which corresponds to the first paragraph in Section 5. 
First, `sim_normal_D.m` was used to compute the layerwise graphical matrix estimation accuracy Table 1 and Tables S.3, S.4 (in the supplement).
The current code is set to $D=4$ latent layers, but can be directly modified to other values of $D$ following the instructions within the script.
Second, `select_K.m` was used to generate Table S.5, which reports the accuracy for estimating the latent dimension in three-latent-layer DDEs.

## Overview
This folder contains proposed algorithms (Algorithm 1, 2 and 6 in the main paper) for each response type (Bernoulli, Poisson, Normal).
The naming convention of the files are as follows:
1. `xxx_init` corresponds to the double-svd initialization in Algorithm 1.
2. `get_SAEM_xxx` corresponds to the stochastic approximation EM in Algorithm 2.
3. `get_EM_xxx` corresponds to the penalized EM in Algorithm 6.
4. The remaining files (F_1, Fun_1, ...) are auxiliary objective functions (expected log-likelihood or its approximation) that are used for M-steps in the EM algorithms.

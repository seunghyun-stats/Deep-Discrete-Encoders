### Description of the 20 newsgroups dataset
Following the preprocessing details in Supplement S.5.1, we have processed the train/test dataset. Each file contains the following information:
1. `X_train.csv` contains the 5883 x 653 data matrix, where the $(i,j)$ th entry counts the number of occurrences of the $j$ th word in document $i$.
2. `lab_train.csv` contains the 5883 x 2 held-out category information (in numerical values) corresponding to each row of `X_train.csv`. This information was used to assign class names to each latent dimension.
3. `train.map` contains the detailed category information corresponding to the numerical values in `lab_train.csv` (also see Fig. S.12).
4. `vocab_train.csv` is a length 653 cell, where each value corresponds to the column name (word) in `X_train.csv`.

The files in the `test dataset` sub-folder are named analogously, where the number of samples/rows in `X_test.csv` is $N = 3320$. The `X_test_subset.csv` and `X_valid_subset.csv` files are the 80%-20% split of `X_test.csv` that is used to compute test perplexity as detailed in Supplement S.5.4.

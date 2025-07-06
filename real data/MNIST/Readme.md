## Overview
The dataset consists of five `.csv` files. The original dataset is publicly available (e.g, see https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/).

First, `MNIST_x_test.csv` and `MNIST_x_train.csv` contains the preprocessed binary data matrices, where each has dimensions $20,679 \times 264$ and $4,157 \times 264$. The first is the train set, and the second is the test set.
Each row corresponds to an image, and each column corresponds to a pixel in an original $28 \times 28$ grid.

Second, `MNIST_lab_test.csv` and `MNIST_lab_train.csv` are length $20,679$/$4,157$ vectors containing the train/test labels. This information is held-out while fitting the model.

Third, `pixel_coord.csv` is a $264 \times 2$ matrix where each row describes the $(x,y)$-coordinate (in the original $28 \times 28$ grid) of the corresponding column index (of the data matrix).

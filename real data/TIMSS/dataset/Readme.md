## Overview
The dataset consists of three `.csv` files.
First, `TIMSS.csv` is the preprocessed (as detailed in Supplement S.5.1) $435 \times 58$ main data matrix. Here, each row corresponds to each test-taking student. 
The first 29 columns correspond to the binary response accuracy of the 29 questions, and the last 29 columns correspond to the continuous response time.

Second, `Q.csv` is a $29 \times 7$ binary matrix that describes the provisional Q-matrix (or equivalently, the matrix $G^{(1)}$ using the notations in the paper) that encodes the graphical structure between the $K^{(1)} = 7$ latent skills and the 29 questions.
Note that for the analysis, we have assumed a common Q-matrix for both the response accuracy and time.

Third, `survey_response.csv` is a categorical vector of length $435$, where each entry is a student's survey response (taking values 1 through 4) to the question "Mathematics is one of my favorite subjects".

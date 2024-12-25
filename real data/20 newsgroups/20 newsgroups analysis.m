%% load data
X = readmatrix("X_train.csv");
vocab = readcell("vocab_train.csv");
category = readmatrix("lab_train.csv");

cat_1 = category(:,2);
cat_2 = category(:,1);

% map stores the sub-category names (comp, rec, sci, pol)
map = importdata('C:\Users\leesa\Documents\Columbia\Research\DDE\20news-bydate-matlab\train.map');
map = map.textdata;

[N, J] = size(X);

%% Fit the 2-latent-layer DDE
% Step 1: select K1 using the spectral-gap estimator
X_trunc = max(X, 0.1);
X_inv = log(X_trunc);
[~, S, ~] = svd(X_inv, "econ"); 
eigval = diag(S);

ratio = zeros(29, 1);
for k = 2:30
    ratio(k-1) = eigval(k)/eigval(k+1)
end
plot(5:20, ratio(5:20), 'b');
hold on; 

highlightX = 8;           
highlightY = ratio(8);
plot(highlightX, highlightY, 's', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');
xlabel('k1');
ylabel('Ratio of eigenvalues');
print('-r300', 'newsgroup_spectral', '-dpng')

% Step 2: spectral initialization
K1 = 8; K2 = 2; % K2 = 3
epsilon = 0.001;         
[prop_ini, B1_ini, B2_ini, A1_ini, A2_ini] = Poi_init_data(X, K1, K2, epsilon);

% Step 3: penalized SAEM / PEM
% below is a one-line code for a fast output
% note that the latent variables are subject to label permutation
[prop, B1, B2, phi, loglik, itera] = get_EM_poisson_data(X, prop_ini, B1_ini, ...
    B2_ini, N^(2/8), N^(2/8), 2*N^(-3/8), 20);


%% uncomment below for the actual implementation %%
% % select tuning parameter via EBIC
% lambda_1_vec = [N^(1/8) N^(2/8) N^(3/8)];
% lambda_2_vec = lambda_1_vec;
% tau_vec = 2 * N.^(-[1/8 2/8 3/8]);
% 
% C = 1;          % number of SAEM samples
% n_rep = 3^3;    % number of replications
% 
% prop_vec = zeros(K2, n_rep); B1_vec = zeros(J, K1+1, n_rep); B2_vec = zeros(K1, K2+1, n_rep);
% A1_vec = zeros(N, K1, n_rep); A2_vec = zeros(N, K2, n_rep); loglik = zeros(n_rep, 1);
% parfor(c = 1:n_rep, 4)
%     kk = fix((c-1)/9)+1; d = rem((c-1), 9);
%     ii = fix(d/3)+1; jj = rem(d, 3)+1;
%     
%     [prop, B1_vec(:,:,c), B2_vec(:,:,c), A1_vec(:,:,c), A2_vec(:,:,c), ~]= get_SAEM_poisson(X, prop_ini, B1_ini, ...
%         B2_ini, lambda_1_vec(ii), lambda_2_vec(jj), tau_vec(kk), A1_ini, A2_ini, C);
%     prop_vec(:,c) = prop;
%     loglik(c) = compute_likelihood(X, prop, B1_vec(:,:,c), B2_vec(:,:,c));
% end
% 
% EBIC = zeros(n_rep, 1); max_val = K2 + J*(K1 + 1) + K1*(K2 + 1); gamma = 1;
% for c = 1:n_rep
%     B1 = B1_vec(:,:,c); B2 = B2_vec(:,:,c);
%     kk = fix((c-1)/4)+1;
%     df = K2 + J + sum(abs(B1(:, 2:end)) > tau_vec(kk), 'all') + ...
%                 K1 + sum(abs(B2(:, 2:end)) > tau_vec(kk), 'all') + J;
%     EBIC(c) = -2*loglik(c) + 2*df*log(N) + 2*gamma*(max_val*log(max_val)-df*log(df)-(max_val-df)*log(max_val-df));
% end
% 
% [~, c] = min(EBIC);
% kk = fix((c-1)/9)+1; d = rem((c-1), 9); ii = fix(d/3)+1; jj = rem(d, 3)+1;
% prop = prop_vec(:,c)'; B1 = B1_vec(:,:,c); B2 = B2_vec(:,:,c);
% 
% [prop, B1, B2, phi, loglik, itera] = get_EM_poisson_data(X, prop, B1, ...
%     B2, lambda_1_vec(ii), lambda_2_vec(jj), tau_vec(kk), tol);
% 
% save("Newsgroup_"+K1".mat")


%% Analysis
% find representative words for each topic
nonzero_word = find(sum(B1_est(:, 2:end),2) ~= 0);
zero_word = find(sum(B1_est(:, 2:end),2) == 0);

top_N_words = 15;
anchor_words = strings([K1, top_N_words]); anchor_index = zeros(K1, top_N_words);
for k = 1:K1
    tmp = zeros(J,1);
    for j = 1:J
        tmp(j) = max(0, min(B1(j,k+1) - B1(j,setdiff(2:(K1+1), k+1)))); 
        % based on largest difference between other row values
    end
    [~, sort_ind_k] = sort(tmp, "descend");
    anchor_words(k, 1:top_N_words) = vocab(sort_ind_k(1:top_N_words));
    anchor_index(k, 1:top_N_words) = sort_ind_k(1:top_N_words);
end

anchor_words

% compute similarity
similarity = 0;
for k = 1:8
    for l = (k+1):8
        commonElements = intersect(anchor_words(k,:), anchor_words(l,:));
        similarity = similarity + length(commonElements);
    end
end

% compute coherence
X_final = readmatrix("X_test.csv");
top_N = 15;
coherence = 0; epsilon = 1;
for k = 1:8
    for v1 = 1:(top_N-1)
        for v2 = (v1+1):top_N
            f_v1 = sum((X_final(:, anchor_index(k, v1)) > 0 ) & (X_final(:, anchor_index(k, v2)) > 0));
            f_v2 = sum(X_final(:, anchor_index(k, v2)) > 0);
            coherence = coherence + log((f_v1 + epsilon)/f_v2);
        end
    end
end
neg_coherence = - coherence/K1

% compute test perplexity
X_final = readmatrix("X_test_subset.csv");  % 80% of words, used to estimate the latent variables
X_valid = readmatrix("X_valid_subset.csv"); % 20% of words, used to evaluate perplexity

[N_fin, J] = size(X_final);
phi = zeros(N_fin, 2^K1, 2^K2); phi_tmp = phi;
phi_2 = zeros(N_fin,1);

A2 = binary(0:(2^K2-1), K2);
A2_long = [ones(2^K2,1), A2];
A1 = binary(0:(2^K1-1), K1);
A1_long = [ones(2^K1,1), A1];

lambda = A1_long * B1';
for i = 1:N_fin
    for b = 1:2^K2
        eta = A2_long(b,:) * B2';
        phi_tmp(i, :, b) = (-sum(exp(lambda), 2) + lambda*X_final(i,:)' + A1*eta' ...
            - sum(log(1+exp(eta))) + log(prop) * A2(b,:)' + log(1 - prop) * (1-A2(b,:))');
    end
    [phi(i,:,:), phi_2(i)] = compute_logistic(reshape(phi_tmp(i,:,:),2^K1,2^K2));
end

pi_1 = zeros(N_fin, K1);
pi = sum(phi, 3);
for i = 1:N_fin
    [~, I] = max(pi(i,:));
    pi_1(i,:) = binary(I-1, K1);
end

lambda = exp([ones(N_fin,1), pi_1] * B1');
lambda_sum = sum(lambda, 2);
lambda_resaled = lambda ./ repmat(lambda_sum, 1, J);
neg_log_perplexity = sum(X_valid.*log(lambda_resaled),'all')/sum(X_valid, 'all');
exp(-neg_log_perplexity)

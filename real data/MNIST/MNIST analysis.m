%% load data
Y = readmatrix("MNIST_x_train.csv");
lab = readmatrix("MNIST_lab_train.csv");

[N, J] = size(Y);

%% Fit the 2-latent-layer DDE
% Step 1: select K1 using the spectral-gap estimator
eps = 0.01
X_trunc = max(Y, eps);
X_trunc = min(X_trunc, 1 - eps);
X_inv = logit(X_trunc);

[~, S, ~] = svd(X_inv, "econ");
eigval = diag(S);

ratio = zeros(29, 1);
for k = 2:30
    ratio(k-1) = eigval(k)/eigval(k+1);
end

plot(3:20, ratio(3:20), 'b');
hold on; 
highlightX = 5;
highlightY = ratio(5);
plot(highlightX, highlightY, 's', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');
xlabel('k1');
ylabel('Ratio of eigenvalues');
% print('-r300', 'mnist_spectral_binarize', '-dpng')

% Step 2: spectral initialization
K1 = 5; K2 = 2;
epsilon = 0.0001;
[prop_ini, B1_ini, B2_ini, A1_ini, A2_ini] = Ber_init_data(Y, K1, K2, epsilon);

% Step 3: penalized SAEM / PEM
C = 1;          % number of SAEM samples
n_rep = 3^3;    % number of replications

prop_vec = zeros(K2, n_rep); B1_vec = zeros(J, K1+1, n_rep); B2_vec = zeros(K1, K2+1, n_rep);
A1_vec = zeros(N, K1, n_rep); A2_vec = zeros(N, K2, n_rep); loglik = zeros(n_rep, 1);
parfor(c = 1:n_rep, 4)
    kk = fix((c-1)/9)+1; d = rem((c-1), 9);
    ii = fix(d/3)+1; jj = rem(d, 3)+1;
    
    [prop, B1_vec(:,:,c), B2_vec(:,:,c), A1_vec(:,:,c), A2_vec(:,:,c), ~]= get_SAEM_bernoulli(Y, prop_ini, B1_ini, ...
        B2_ini, lambda_1_vec(ii), lambda_2_vec(jj), tau_vec(kk), A1_ini, A2_ini, C, 500);
    prop_vec(:,c) = prop;
    loglik(c) = compute_likelihood(Y, prop, B1_vec(:,:,c), B2_vec(:,:,c));
end

EBIC = zeros(n_rep, 1); max_val = K2 + J*(K1 + 1) + K1*(K2 + 1); gamma = 1;
for c = 1:n_rep
    B1 = B1_vec(:,:,c); B2 = B2_vec(:,:,c);
    kk = fix((c-1)/4)+1;
    df = K2 + J + sum(abs(B1(:, 2:end)) > tau_vec(kk), 'all') + ...
                K1 + sum(abs(B2(:, 2:end)) > tau_vec(kk), 'all') + J;
    EBIC(c) = -2*loglik(c) + 2*df*log(N) + 2*gamma*(max_val*log(max_val)-df*log(df)-(max_val-df)*log(max_val-df));
end

[~, c] = min(EBIC);
kk = fix((c-1)/9)+1; d = rem((c-1), 9); ii = fix(d/3)+1; jj = rem(d, 3)+1;
prop = prop_vec(:,c)'; B1 = B1_vec(:,:,c); B2 = B2_vec(:,:,c);

[prop, B1, B2, phi, loglik, itera] = get_EM_poisson_data(X, prop, B1, ...
    B2, lambda_1_vec(ii), lambda_2_vec(jj), tau_vec(kk), tol);


%% visualizing basis image: B1
basis = zeros(28, 28, K1+1);
for k = 1:K1+1
    for j = 1:J
        x = pixel_coord(j,:);
        basis(x(1), x(2), k) = B1_fin(j, k);
    end
end

for k = 1:(K1+1)
    h =  heatmap(thres(min(basis(:, :, k), 0), 1.5), 'Colormap', cmap);
    grid off; caxis([-5, 5]);
    Ax = gca;
    Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
    Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
    % print('-r300', "EM_basis_k="+(k-1)+"_negative", '-dpng')
end


%% image reconstruction
A1_fin = A1_vec(:,:,c); A2_fin = A1_vec(:,:,c);
pi = sum(phi, 3); pii = reshape(sum(phi, 2), N, 2^K2);

for i = 1:N
    [~, I] = max(pi(i,:));
    [~, II] = max(pii(i,:));
    A1_fin(i,:) = binary(I-1, K1);
    A2_fin(i,:) = binary(II-1, K2);
end

imagesc([lab/(L-1), A1_fin], [0 1]);
imagesc([lab/(L-1), A2_fin], [0 1]);

eta = logistic([1, A1_fin(index,:)]*B1_fin');

recon = zeros(28, 28);
recon_ini = zeros(28, 28);
for j = 1:J
    x = pixel_coord(j,:);
    recon(x(1), x(2)) = eta(j);
    recon_ini(x(1), x(2)) = eta_ini(j);
end

true = zeros(28, 28);
for j = 1:J
    x = pixel_coord(j,:);
    true(x(1), x(2)) = Y(index, j);
end

h1=heatmap(true+0.0001, 'Colormap', cmap_recon); caxis([0, 1])
Ax = gca;
grid off; colorbar off;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
% h1.FontSize = 14;
print('-r300', "true_"+digit, '-dpng')

h2=heatmap(recon, 'Colormap', cmap_recon); caxis([0, 1])
Ax = gca;
grid off; colorbar off;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
h3.FontSize = 14;
print('-r300', "recon_fin_"+digit, '-dpng')

% compute reconstruction error (among the J pixels)
err_fin = zeros(N,1);

for i = 1:N
    eta_true = Y(i,:);
    eta_fin = logistic([1, A1_fin(i,:)]*B1_fin');

    err_fin(i) = mean((eta_fin > 0.5) == eta_true);
end
mean(err_fin)


%% Classification
est_0 = find(A1_fin(:,1) == 1);
est_1 = find(A1_fin(:,1) == 0 & A1_fin(:,5) == 0 & (A1_fin(:,4) == 0 | (A1_fin(:,4) == 1 & A1_fin(:,3) == 0)));
est_2 = find(A1_fin(:,1) == 0 & A1_fin(:,5) == 1 & (A1_fin(:,4) == 0 | A1_fin(:,2) == 1));

lab_est = ones(N,1)*3;
lab_est(est_0) = 0; lab_est(est_1) = 1; lab_est(est_2) = 2;
mean(lab_est == lab)

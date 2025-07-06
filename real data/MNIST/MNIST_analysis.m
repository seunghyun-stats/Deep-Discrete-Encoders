addpath('dataset')
addpath('supplementary functions')
addpath('Deep-Discrete-Encoders-main\main algorithms\Bernoulli')

%% load data
Y = readmatrix("MNIST_x_train.csv");
lab = readmatrix("MNIST_lab_train.csv");
pixel_coord = readmatrix("pixel_coord.csv");
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

% generate the left panel of Fig. S.13
plot(3:20, ratio(3:20), 'b');
hold on; 
highlightX = 5;
highlightY = ratio(5);
plot(highlightX, highlightY, 's', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');
xlabel('k1');
ylabel('Ratio of eigenvalues');
print('-r300', 'mnist_spectral_binarize', '-dpng') 


% Step 2: spectral initialization
K1 = 5; K2 = 2;
epsilon = 0.0001;
[prop_ini, B1_ini, B2_ini, A1_ini, A2_ini] = Ber_init_data(Y, K1, K2, epsilon);


% Step 3: penalized EM
% the following is a simplification of the actual implementation, which
% produces similar results
[prop, B1, B2, A1, A2, ~]= get_SAEM_bernoulli(Y, prop_ini, B1_ini, ...
         B2_ini, N.^(4/8*0.9), N.^(2/8*0.9)/2, 2 * N.^(3/8*0.9-1/2), A1_ini, A2_ini, C, 10);

% compute conditional probabilities
phi = zeros(N, 2^K1, 2^K2); A1_all = binary(0:31,5); A2_all = binary(0:3,2);
for a = 1:2^K1
    lambda = [1, A1_all(a,:)] * B1';
    for b = 1:2^K2
        eta = [1, A2_all(b,:)] * B2';
        phi(:, a, b) = exp(sum(Y.*lambda, 2) - sum(log(1+exp(lambda))) + sum(A1_all(a,:).*eta) ...
            - sum(log(1+exp(eta))) + log(prop) * A2_all(b,:)' + log(1 - prop) * (1-A2_all(b,:))');
    end
end
phi_2 = sum(sum(phi, 3), 2);
phi = phi ./ repmat(phi_2, 1, 2^K1, 2^K2);


%%%%% the following code was actually used for our analysis to select
%%%%% tuning parameters, but is commented out for faster illustration.
%%%%% the exact output can be acessed by loading the separate .mat file
% C = 1;          % number of samples
% n_rep = 3^3;    % number of replications
% 
% prop_vec = zeros(K2, n_rep); B1_vec = zeros(J, K1+1, n_rep); B2_vec = zeros(K1, K2+1, n_rep);
% A1_vec = zeros(N, K1, n_rep); A2_vec = zeros(N, K2, n_rep); loglik = zeros(n_rep, 1);

% tol=3;
% const = 0.9;
% % candidate tuning parameter values
% lambda_1_vec = N.^([2/8 3/8 4/8]*const);
% lambda_2_vec = lambda_1_vec/2;
% tau_vec = 2 * N.^([2/8 3/8 4/8]*const-1/2);
% 
% parfor(c = 1:n_rep, 4)
%     kk = fix((c-1)/9)+1; d = rem((c-1), 9);
%     ii = fix(d/3)+1; jj = rem(d, 3)+1;
%     
%     [prop, B1_vec(:,:,c), B2_vec(:,:,c), A1_vec(:,:,c), A2_vec(:,:,c), ~]= get_SAEM_bernoulli(Y, prop_ini, B1_ini, ...
%         B2_ini, lambda_1_vec(ii), lambda_2_vec(jj), tau_vec(kk), A1_ini, A2_ini, C, 50);
%     prop_vec(:,c) = prop;
%     loglik(c) = compute_likelihood(Y, prop, B1_vec(:,:,c), B2_vec(:,:,c));
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
% [prop, B1, B2, phi, loglik] = get_EM_bernoulli(Y, prop, B1, ...
%     B2, lambda_1_vec(ii), lambda_2_vec(jj), tau_vec(kk), tol);
%%%%% end of commented out code

B1_fin = B1; B2_fin = B2;

%% The following code generates results in the main paper

%% visualize basis images (Table 2)
% re-shape each column of B1
basis = zeros(28, 28, K1+1);
for k = 1:K1+1
    for j = 1:J
        x = pixel_coord(j,:);
        basis(x(1), x(2), k) = B1_fin(j, k);
    end
end

% visualize in the original grid
[~, cmap] = colorbarpzn(-5, 5, 'dft', 'ywg'); 
for k = 1:(K1+1)
    h =  heatmap(basis(:, :, k), 'Colormap', cmap);
    grid off; caxis([-5, 5]);
    Ax = gca;
    Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
    Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
    print('-r300', "EM_basis_k="+(k-1), '-dpng')
end

% visualize negative parts
for k = 1:(K1+1)
    h =  heatmap(thres(min(basis(:, :, k), 0), 1.5), 'Colormap', cmap);
    grid off; caxis([-5, 5]);
    Ax = gca;
    Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
    Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
    print('-r300', "EM_basis_k="+(k-1)+"_negative", '-dpng')
end


%% visualize generated and reconstructed digits (Figures 3 and S.16)
% reconstructed digits
A1_fin = zeros(N,K1); A2_fin = zeros(N,K2);
pi = sum(phi, 3); pii = reshape(sum(phi, 2), N, 2^K2);

for i = 1:N
    [~, I] = max(pi(i,:));
    [~, II] = max(pii(i,:));
    A1_fin(i,:) = binary(I-1, K1);
    A2_fin(i,:) = binary(II-1, K2);
end

index = 1; % choose any index between 1 and N
eta = logistic([1, A1_fin(index,:)]*B1_fin');

recon = zeros(28, 28);
recon_ini = zeros(28, 28);
for j = 1:J
    x = pixel_coord(j,:);
    recon(x(1), x(2)) = eta(j);
end

true = zeros(28, 28);
for j = 1:J
    x = pixel_coord(j,:);
    true(x(1), x(2)) = Y(index, j);
end

[~, cmap_recon] = colorbarpzn(0, 1, 'level', 30, 'colorP', [0.6 0.4 0.3]);

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

% generated digits
A1_example = [1,0,1,0,1]; % take a binary vector of length K1

val = logistic([1 A1_example] *B1_fin');
recon_zero = zeros(28, 28);
for j = 1:J
    x = pixel_coord(j,:);
    true(x(1), x(2)) = val(j);
end

h1=heatmap(true+0.0001, 'Colormap', cmap_recon); caxis([0, 1])
Ax = gca;
grid off; colorbar off;
Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
print('-r300', label+"_type"+"_"+ind, '-dpng')



%% compute error metrics (Table 4, DDE column)
% train classification error
est_0 = find(A1_fin(:,1) == 1);
est_1 = find(A1_fin(:,1) == 0 & A1_fin(:,5) == 0 & (A1_fin(:,4) == 0 | (A1_fin(:,4) == 1 & A1_fin(:,3) == 0)));
est_2 = find(A1_fin(:,1) == 0 & A1_fin(:,5) == 1 & (A1_fin(:,4) == 0 | A1_fin(:,2) == 1));

lab_est = ones(N,1)*3;
lab_est(est_0) = 0; lab_est(est_1) = 1; lab_est(est_2) = 2;
mean(lab_est == lab)

% train reconstruction error
err_fin = zeros(N,1);

for i = 1:N
    eta_true = Y(i,:);
    eta_fin = logistic([1, A1_fin(i,:)]*B1_fin');

    err_fin(i) = mean((eta_fin > 0.5) == eta_true);
end
mean(err_fin)

% load test data
YY = csvread("MNIST_x_test.csv");
lab_test = csvread("MNIST_lab_test.csv");

N_test = size(YY, 1);
phi = zeros(N_test, 2^K1, 2^K2); % 
B1 = B1_fin; B2 = B2_fin; prop = prop_final; 
A1 = binary(0:(2^K1-1), K1); A2 = binary(0:(2^K2-1), K2);

for a = 1:2^K1
    lambda = [1, A1(a,:)] * B1';
    for b = 1:2^K2
        eta = [1, A2(b,:)] * B2';
        phi(:, a, b) = exp(sum(YY.*lambda, 2) - sum(log(1+exp(lambda))) + sum(A1(a,:).*eta) ...
            - sum(log(1+exp(eta))) + log(prop) * A2(b,:)' + log(1 - prop) * (1-A2(b,:))');
    end
end
phi_2 = sum(sum(phi, 3), 2);
phi = phi ./ repmat(phi_2, 1, 2^K1, 2^K2);
pi = sum(phi, 3); pii = reshape(sum(phi, 2), N, 2^K2);

A1_test = zeros(N, K1);
A2_test = zeros(N, K2);
for i = 1:N
    [~, I] = max(pi(i,:));
    [~, II] = max(pii(i,:));
    A1_test(i,:) = binary(I-1, K1);
    A2_test(i,:) = binary(II-1, K2);
end

% test classification error
est_0 = find(A1_test(:,1) == 1);
est_1 = find(A1_test(:,1) == 0 & A1_test(:,5) == 0 & (A1_test(:,4) == 0 | (A1_test(:,4) == 1 & A1_test(:,3) == 0)));
est_2 = find(A1_test(:,1) == 0 & A1_test(:,5) == 1 & (A1_test(:,4) == 0 | A1_test(:,2) == 1));

lab_est_test = ones(N,1)*3;
lab_est_test(est_0) = 0; lab_est_test(est_1) = 1; lab_est_test(est_2) = 2;
mean(lab_est_test == lab_test)

% test reconstruction error
err_fin_test = zeros(N,1);

for i = 1:N_test
    eta_true = Y(i,:);
    eta_fin = logistic([1, A1_fin(i,:)]*B1_fin');
    err_fin_test(i) = mean((eta_fin > 0.5) == eta_true);
end
mean(err_fin_test)



%% visualize latent variable estimates and labels (Figure 5)
pink3 = pink(3); reversepink = [pink3(3,:); pink3(1,:)];

imagesc(A1_fin);
xticks(1:K1)
xlabel('$k^{(1)}$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel("$N = $"+N+" images", 'Interpreter', 'latex', 'FontSize', 20); pbaspect([1 1.5 1])
set(gca, 'FontSize', 18)
yticks([])
% axis off
old2 = colormap([0,1,0;1,0,0]); colormap( flipud(old2) ); 
bar2 = colorbar; bar2.Ticks = [0.25 0.75]; bar2.TickLabels = {'0' '1'};
title('Estimated $\hat{\mathbf{A}}^{(1)}$', 'Interpreter', 'latex', 'FontSize', 20);
print('-r300', 'A1', '-dpng')

imagesc(A2_fin);
xticks(1:K2)
xlabel('$k^{(2)}$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel("$N = $"+N+" images", 'Interpreter', 'latex', 'FontSize', 20); pbaspect([1 1.5 1])
set(gca, 'FontSize', 18);
yticks([])
old3 = colormap(reversepink); colormap( flipud(old3) );
bar3 = colorbar; bar3.Ticks = [0.25 0.75]; bar3.TickLabels = {'0' '1'};
title('Estimated $\hat{\mathbf{A}}^{(2)}$', 'Interpreter', 'latex', 'FontSize', 20);
print('-r300', 'A2', '-dpng')

imagesc(lab_est); 
colormap(turbo(4)); 
% cbh3 = colorbar; set(cbh3,'YTick', 1:3)
xticks(''); xticklabels({''}); % xlabel('estimated digits') % xlabel('held out digits')
ylabel("$N = $"+N+" images", 'Interpreter', 'latex', 'FontSize', 20); pbaspect([1.4 3 1])
yticks(''); yticklabels({''}); 
t = cell(4,1);
t{1} = '0'; t{2} = '1'; t{3} = '2'; t{4} = '3'
text([1 1 1 1], [2467 7772 13095 18129], t, 'HorizontalAlignment', 'Center')
title('Estimated Label'); set(gca, 'FontSize', 12)
print('-r300', 'label_est', '-dpng')

imagesc(lab); 
colormap(turbo(4)); 
set(gca, 'FontSize', 20)
% cbh3 = colorbar; set(cbh3,'YTick', 1:3)
xticks(''); xticklabels({''}); % xlabel('estimated digits') % xlabel('held out digits')
ylabel("$N = $"+N+" images", 'Interpreter', 'latex', 'FontSize', 20); pbaspect([1.4 3 1])
yticks(''); yticklabels({''}); 
t = cell(4,1);
t{1} = '0'; t{2} = '1'; t{3} = '2'; t{4} = '3'
text([1 1 1 1], [2467 7772 13095 18129], t, 'HorizontalAlignment', 'Center', 'FontSize', 20)
title('True Label'); 
print('-r300', 'label_true', '-dpng')


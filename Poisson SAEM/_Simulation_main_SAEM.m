

N = 1000; % change (do N = 1000, 2000, 3000, 4000)

%% Setting 1
K2 = 2;

%% Setting 2
K2 = 4;

%% Setting 3
K2 = 6;

K2 = 8;

%% define parameters (K1 = 3*K2, J = 9*K2)
% discrete parameters

K1 = 3*K2;

G1 = [eye(K1); eye(K1); eye(K1)];
G2 = [eye(K2); eye(K2); eye(K2)];

for k = 1:K1-1
    G1(k+1, k) = 1;
end
[J, ~] = size(G1);

for l = 1:K2-1
    G2(l+1, l) = 1;
end

% continuous parameters
prop_true = 0.5*ones(1, K2);

% B2_sub = [3*eye(K2); 2*eye(K2); ones(1, K2)];
% b2 = [-ones(K2,1); -2*ones(K2,1); -1];
% B2 = [b2, B2_sub];

B2_sub = zeros(K1, K2);
s = sum(G2, 2);
for l = 1:K2
    B2_sub(l,l) = 3; % G1(k,k) .* (K1+2-k)/2;
    B2_sub(l+K2,l) = 2;
    B2_sub(l+2*K2,l) = 2;
end
for l = 1:K2-1
    B2_sub(l+1, l) =  G2(l+1, l)*1.5;
end
B2 = [[-1*ones(K2,1); -1.5*ones(K2,1); -0.5*ones(K2,1)], B2_sub];
% svd(B2)

B1_sub = zeros(J, K1);
s = sum(G1, 2);
for k = 1:K1
    B1_sub(k,k) = 3; % G1(k,k) .* (K1+2-k)/2;
    B1_sub(k+K1,k) = 2;
    B1_sub(k+2*K1,k) = 2;
end
for k = 1:K1-1
    B1_sub(k+1, k) =  G1(k+1, k)*1.5;
end
B1 = [[-1*ones(K1,1); -1.5*ones(K1,1); -0.5*ones(K1,1)], B1_sub];

% vecnorm(B1)
% svd(B1)                         % need full column rank!


C = 40;
n_sample = 1;

% SCAD
lambda_1_vec = 1*N^(1/3); %% CHANGE the exponent of N...
lambda_2_vec = 0.5*N^(1/3);
tau_vec = 7*N^(-1/3);

% lambda_1_vec = [1.2, 0.8]*N^(1/3);   % select smallest from the 5 vals.
% lambda_2_vec = [0.5, 0.25]/2*N^(1/3); % divide lambda_1 by 3??
% tau_vec = [10, 7.5]*N^(-1/3); % chose 0.7

prop_ini = zeros(K2, C);
prop_final = zeros(K2, C);

A1_final = zeros(N, K1, C);
A1_ini = A1_final;
A2_final = zeros(N, K2, C);
A2_ini = A2_final;

B1_final = zeros(J, K1+1, C);
B1_ini = B1_final;
B2_final = zeros(K1, K2+1, C);
B2_ini = B2_final;
loglik_est = zeros(C, 1);
loglik_true = zeros(C, 1);

% perms_K1 = perms(1:K1); % This is large when K1 large..
perms_K2 = perms(1:K2);
err_G1_ini = zeros(C,1);
err_G1_final = zeros(C,1);
err_G2_ini = zeros(C,1);
err_G2_final = zeros(C,1);

time_vec = zeros(C,1);
lambda_select = zeros(C,1);
tau_select = zeros(C,1);
itera = zeros(C,1);

parpool(4)
parfor(c= 1:C, 4)
    rng(c);
    [X, A1_true, A2_true] = generate_X_poisson(N, prop_true, B1, B2);
    % loglik_true(c) = compute_likelihood(X, prop_true, B1, B2);      % this is slow...
    
    tic;
    %% spectral initialization
    %% 1st layer
    epsilon = 0.0001;         % small constant, 10^(-4) in Chen 2019
    [U,S,V] = svd(X);
    m = max(K1+1, sum(diag(S) > 1.01*sqrt(N))); % K1+1 if we include intercept
    
    X_top_m = U(1:end, 1:m) * S(1:m, 1:m) * V(1:end, 1:m)';
    X_trunc = max(X_top_m, epsilon);
    X_inv = log(X_trunc);
    X_inv_adj = X_inv- ones(N,1) * mean(X_inv);
    % plot(svd(X_inv_adj))

    [U_true_adj, ~, V_true_adj] = svd(X_inv_adj); 
    [R_V, ~] = rotatefactors(V_true_adj(:,1:K1), 'method', 'varimax');
    B1_est = thres(R_V, 0.15); % column norm = 1
    %% Change 0.15 to 1/2/sqrt(J)
    % B1_est = thres(R_V, 1/2/sqrt(J));

    % adjust sign flip
    Sign_flip_multiplier_B = [repmat(2*(mean(B1_est(:,1:end), 1) > 0)-1, J, 1)];
    B1_est = B1_est .* Sign_flip_multiplier_B;
    % B1_est = max(B1_est, 0); % can also do this
    G1_est = double(B1_est ~=0);
    
    % adjust column rotation (current method fails for large K1)
    col_perm = zeros(1, K1);
    remaining_cols = 1:K1;
    for k = 1:K1
        [~, tmp] = max(sum(G1_est([k, K1+k, 2*K1+k], remaining_cols), 1)); % find
        % index = find(remaining_cols == col_perm(k));
        col_perm(k) = remaining_cols(tmp);
        remaining_cols(tmp) = [];
    end
    B1_est = B1_est(:,col_perm);

%     error = zeros(K1,1);
%     for k1 = 1:size(perms_K1,1)
%         ind = perms_K1(k1,:);
%         B1_perm = B1_est(:, ind);
%         G1_perm = (B1_perm ~= 0);
%         error(k1) = mean(G1_perm == G1, 'all');
%     end
%     [err_G1_ini(c), perm_ind] = max(error);
%     B1_est = B1_est(:, perms_K1(perm_ind,:));
    % G1_est = double(B1_est ~=0);
    err_G1_ini(c) = mean((B1_est ~= 0) == G1, 'all');
    
    % estimate A1 (binary)
    A1_est_nonbin = X_inv_adj * B1_est * inv(B1_est' * B1_est);
    A1_est = double(A1_est_nonbin > 0);
    G1_est = (B1_est ~= 0);

    % re-estimate B1
    A1_centered = A1_est - ones(N,1) * mean(A1_est, 1);
    B1_re_est = (inv(A1_centered' * A1_centered) * A1_centered' * X_inv_adj)'/2;    % dividing by 2 is artificial
    B1_re_est = B1_re_est .* G1_est; % thres(B1_re_est, tau);
    b1 = mean(X_inv_adj, 1)' - B1_re_est * mean(A1_est, 1)';
    % or simply do
    % b1 = -ones(J,1);

    %% 2nd layer
    % [U,S,V] = svd(A1_est_nonbin);  %% normalize??
    [U,S,V] = svd(A1_est);
    m = max(K2+1, sum(diag(S) > 1.01*sqrt(N)));
    A1_top_m = U(1:end, 1:m) * S(1:m, 1:m) * V(1:end, 1:m)';
    A1_trunc = max(min(A1_top_m, 1-epsilon), epsilon);

    A_inv = logit(A1_trunc);
    A_inv_adj = A_inv - ones(N,1)*mean(A_inv);
    
    [~, D, V_adj] = svd(A_inv_adj); 
    [R_V, ~] = rotatefactors(V_adj(:,1:K2), 'method', 'varimax');
    
    B2_est = thres(R_V, 0.15);
    
    % adjust sign flip
    Sign_flip_multiplier_B = [repmat(2*(mean(B2_est(:,1:end), 1) > 0)-1, K1, 1)];
    B2_est = B2_est .* Sign_flip_multiplier_B;
    
    % adjust column permutation
    error = zeros(K2,1);
    for k2 = 1:size(perms_K2,1)
        ind = perms_K2(k2,:);
        B2_perm = B2_est(:, ind);
        G2_perm = (B2_perm ~= 0);
        error(k2) = mean(G2_perm == G2, 'all');
    end
    [err_G2_ini(c), perm_ind] = max(error);
    B2_est = B2_est(:, perms_K2(perm_ind, :));
    G2_est = double(B2_est ~=0);
    
    A2_est = A_inv_adj * B2_est * inv(B2_est' * B2_est);
    A2_est = double(A2_est > 0);
    % mean(A2_est == A2_true)
    
    % estimate b2
    A2_centered = A2_est - ones(N,1) * mean(A2_est, 1);
    B2_re_est = (inv(A2_centered' * A2_centered) * A2_centered' * A_inv_adj)'/2;    % dividing by 2 is artificial
    B2_re_est = B2_re_est .* G2_est; % thres(B2_re_est, tau);
    b2 = mean(A_inv_adj, 1)' - B2_re_est * mean(A2_est, 1)';
    % b2 = - ones(K2, 1);

    prop_ini(:,c) = mean(A2_est > 0);    % this is actually quite good!
    B1_ini(:,:,c) = [b1, B1_re_est];
    B2_ini(:,:,c) = [b2, B2_re_est];
    
    A1_ini(:,:,c) = A1_est;
    A2_ini(:,:,c) = A2_est;
    
    %% EM
%     BIC_vec = zeros(length(lambda_1_vec), length(lambda_2_vec), length(tau_vec));
%     for ii = 1:length(lambda_1_vec)
%         for jj = 1:length(lambda_2_vec)
%             for kk = 1:length(tau_vec)
%                 [prop_0, B1_0, B2_0, ~, ~, ~] = get_SAEM_poisson( ...
%                     X, prop_ini(:,c)', B1_ini(:,:,c), B2_ini(:,:,c), lambda_1_vec(ii), ...
%                     lambda_2_vec(jj), tau_vec(kk), A1_est, A2_est, n_sample);
%                 df = K2 + J + sum(abs(B1_0(:, 2:end)) > tau_vec(kk), 'all') + ...
%                     K1 + sum(abs(B2_0(:, 2:end)) > tau_vec(kk), 'all');
%                 loglik = compute_likelihood(X, prop_0, B1_0, B2_0);     % slow..
%                 BIC_vec(ii, jj, kk) = -2*loglik + 2*df*log(N);          % EBIC: + 2*log(nchoosek_prac(max, df))
%             end
%         end
%     end
%     
%     [~, I] = min(BIC_vec(:));
%    [II, JJ, KK] = ind2sub([length(lambda_1_vec) length(lambda_2_vec) length(tau_vec)], I);
    II = 1; JJ = 1; KK = 1;
    [prop_final(:,c), B1_fin, B2_fin, A1_final(:,:,c), A2_final(:,:,c), itera(c)] = get_SAEM_poisson( ...
                    X, prop_ini(:,c)', B1_ini(:,:,c), B2_ini(:,:,c), lambda_1_vec(II), ...
                    lambda_2_vec(JJ), tau_vec(KK), A1_est, A2_est, n_sample);
    
    B1_final(:,:,c) = [B1_fin(:, 1), thres(B1_fin(:,2:end), tau_vec(KK))];
    B2_final(:,:,c) = [B2_fin(:, 1), thres(B2_fin(:,2:end), tau_vec(KK))];
    time_vec(c) = toc;

    lambda_1_select(c) = lambda_1_vec(II);
    lambda_2_select(c) = lambda_2_vec(JJ);
    tau_select(c) = tau_vec(KK);
    % loglik_est(c) = compute_likelihood(X, prop_final(:,c)', B1_final(:,:,c), B2_final(:,:,c));
    fprintf('%d-th iteration complete \n', c);
end

mean(time_vec)
mean(itera)-1

ind = (1:C);
ind = ind(loglik_est ~= 0)

mean(loglik_est > loglik_true)
[loglik_est, loglik_true]

mse_prop = 0;
mse_B1 = 0;
mse_B2 = 0;
err_G1 = zeros(C,1);
err_G2 = zeros(C,1);

% tau = tau_vec(2);
for c = ind
    %B1_est = [B1_final(:, 1, c), thres(B1_final(:,2:end,c), tau)];
    %B2_est = [B2_final(:, 1, c), thres(B2_final(:,2:end,c), tau)];
    B1_est = B1_final(:, :, c);
    B2_est = B2_final(:, :, c);

    err_G1_final(c) = mean(G1 == (B1_est(:,2:end) ~= 0), 'all');
    err_G2_final(c) = mean(G2 == (B2_est(:,2:end) ~= 0), 'all');

    mse_prop = mse_prop + mean((prop_final(:,c)' - prop_true).^2);
    mse_B1 = mse_B1 + mean((B1_est - B1).^2, 'all');
    mse_B2 = mse_B2 + mean((B2_est - B2).^2, 'all');
end

mse_prop = mse_prop / length(ind);
mse_B1 = mse_B1 / length(ind);
mse_B2 = mse_B2 / length(ind);

acc_G1 = mean(err_G1_ini(ind))
acc_G2 = mean(err_G2_ini(ind))

mean(err_G1_final(ind))
mean(err_G2_final(ind)) %% this is very low

save("_Poisson_SAEM_"+N+"_"+K2+'.mat')

sqrt(mse_prop)
sqrt(mse_B1)
sqrt(mse_B2)

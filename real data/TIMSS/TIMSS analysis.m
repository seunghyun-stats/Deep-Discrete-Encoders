%% data preprocessing
K2 = 1; % K2 = 2;
K1 = 7;
J = 29;

G1 = importdata("Q.csv").data;
Y = importdata("time.csv");

N = size(Y,1);

X = log(Y);
row_nonmissing = sum(isnan(X), 2) == 0;
X_adj = X(row_nonmissing, :);
X = X_adj;


%% main estimation
C = 30;
B10 = zeros(J, K1+1, C);
B20 = zeros(K1, K2+1, C); 
gamma_0 = zeros(J, C); prop_0 = zeros(K2,C);
loglik_vec = zeros(C,1);
tol = 1;

K2 = 1; G2 = ones(K1,1); 
B20 = zeros(K1, K2+1, C); prop_0 = zeros(K2,C);
parfor(c = 1:C, 4)
    rng(c);
    prop_in = 0.2 + 0.4*rand(1,K2);
    gamma_in = 0.4*ones(J,1);
    B1_in = [1.5*ones(J,1)+rand(J,1), G1.*(rand(J, K1)+0.5)];
    B2_in =[-2*ones(K1,1)+rand(K1,1), G2.*(2*rand(K1, K2)+0.5)];
    
    % [prop_in, B1_in, B2_in, gamma_in, A1_in, A2_in, ...
    %    ~, ~] = Normal_init_data(X_adj, K1, K1, K2, K2, G1, G2, 0.01);

    [prop, B1, B2, gamma, loglik, itera] = get_EM_normal_confirm(X, prop_in, B1_in, B2_in, gamma_in, G1, G2);

    prop_0(:,c) = prop; B10(:,:,c) = B1; B20(:,:,c) = B2; gamma_0(:,c) = gamma;
    loglik_vec(c) = loglik;
end

[~,c] = max(loglik_vec);
prop = prop_0(:,c) 
B1 = B10(:,:,c)
B2 = B20(:,:,c) 
gamma = gamma_0(:,c)

df = K2 + J + sum(G1, 'all') + sum(G2, 'all');
BIC = -2*max(loglik_vec) + 2*df*log(N);


%% compute phi and estimate the latent variables
phi = zeros(N, 2^K1, 2^K2);

A2 = binary(0:(2^K2-1), K2); A2_long = [ones(2^K2,1), A2];
eta_1 = B1*A1_long';
eta_2 = B2*A2_long';

for i = 1:N
    exponent = zeros(2^K1, 2^K2);
    for a = 1:2^K1
        for b = 1:2^K2
            exponent(a,b) = -sum((X(i,:)'- eta_1(:,a)).^2./gamma)/2 + A1(a,:) * eta_2(:,b) - sum(log(1+exp(eta_2(:,b)))) ...
                            + log(prop ./ (1-prop)) * A2(b,:)';
        end
    end
    phi(i, :, :) = exp(exponent - max(max(exponent)));
    tmp = sum(phi(i, :, :), 'all');
    phi(i,:,:) = phi(i,:,:)/tmp;
end

% A2
psi = zeros(N, 2^K2); A2_est = zeros(N, K2);
for i = 1:N
    for b = 1:2^K2
        psi(i, b) = sum(phi(i,:,b), "all");
    end
end

for i = 1:N
    [~, b] = max(psi(i,:));
    A2_est(i,:) = binary(b-1, K2);
end
mean(A2_est, 1);
save("A2_est.csv", A2_est)


% A1
psi_a = zeros(N, 2^K1); A1_est = zeros(N, K1);
for i = 1:N
    for a = 1:2^K1
        psi_a(i, a) = sum(phi(i,a,:), "all");
    end
end

for i = 1:N
    [~, b] = max(psi_a(i,:));
    A1_est(i,:) = binary(b-1, K1);
end
mean(A1_est, 1);


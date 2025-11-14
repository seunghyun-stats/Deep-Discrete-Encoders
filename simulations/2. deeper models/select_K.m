% This code was used to generate Table S.5.

addpath('utilities')
addpath('Auxiliary code')

D = 3; K3 = 2;

%% define parameters
% coefficients B_s
max_val = 5; 
K2 = 3*K3; K1 = 3*K2; J = 3*K1;
B3 = generate_B(K2,K3,max_val); G3 = B3(:,2:end) ~= 0;
B2 = generate_B(K1,K2,max_val); G2 = B2(:,2:end) ~= 0;
B1 = generate_B(J,K1,max_val); G1 = B1(:,2:end) ~= 0;

B_cell_true = cell(D,1); 
B_cell_true{1} = B1; B_cell_true{2} = B2; B_cell_true{3} = B3;

K_cell = cell(D,1);
K_cell{1} = K1; K_cell{2} = K2; K_cell{3} = K3;

prop_true = 0.5*ones(1, K_cell{D}); 
gamma = ones(J,1);


%% main simulation for selecting the number of latent variables per layer
K1_min = floor(J/4)+1; K1_max = J/2; K1_grid = K1_min:K1_max;

C = 400; % number of scenarios
epsilon = 0.0001; % threshold for spectal decomposition
n_vec = [500 1000 2000 4000 8000 16000];

K_est = cell(C); % this variable stores the estimated latent dimensions
for c = 1:C
    K_est{c} = zeros(length(n_vec), D);
end
time_vec = zeros(length(n_vec),C);

for a = 1:length(n_vec)
    N = n_vec(a);
    
    parfor(c= 1:C, 4)
        rng(50+c);
        [X, A_cell_true] = generate_X_normal_D(N, prop_true, D, B_cell_true, gamma);
        tic;
   
        % select K1
        X_centered = X- ones(N,1) * mean(X);
        eigval = svd(X_centered, "econ");
        
        ratio_1 = zeros(K1_max,1);
        for k = K1_grid
            ratio_1(k) = eigval(k)/eigval(k+1);
        end
        [~, K_select] = max(ratio_1);
        K_est{c}(a,1) = K_select;

        A_est = Normal_init_1(X, K_select, B_cell_true{1});
        
        % select K_d for d >= 2
        for d = 2:D
            K_d_min = max(ceil(K_select/4), 2); K_d_max = min(floor(K_select/2)); 
            K_d_grid = K_d_min:K_d_max;    
            ratio_2 = zeros(K_d_max,1);

            [U,S,V] = svd(A_est, "econ");
            m = max(K_d_max+1, sum(diag(S) > 1.01*sqrt(N)));
            A_top_m = U(1:end, 1:m) * S(1:m, 1:m) * V(1:end, 1:m)';
            A_trunc = max(min(A_top_m, 1-epsilon), epsilon);
            A_inv = logit(A_trunc);
            A_inv_adj = A_inv - ones(N,1)*mean(A_inv);
            
            eigval = svd(A_inv_adj, 'econ'); 
            for k = K_d_grid
                ratio_2(k) = eigval(k)/eigval(k+1);
            end
            [~, K_select] = max(ratio_2);
            K_est{c}(a, d) = K_select;

            if d < D
                A_est = binary_init_K(N, size(A_est, 2), K_select, A_est, 2, epsilon);
            end
        end
        time_vec(a,c) = toc;
        fprintf('%d-th iteration complete \n', c);
    end
end
save("Select_K_D=" + D +".mat");

acc_K = zeros(D,length(n_vec));
K_est_array = zeros(C,length(n_vec),D);
for d = 1:D
    for n = 1:length(n_vec)
        for c = 1:C
            K_est_array(c,n,d) = K_est{c}(n,d);
        end
    end
end

for d = 1:D
    for n = 1:length(n_vec)
        acc_K(d,n) = mean(K_est_array(:,n,d) == K_cell{d})*100;
    end
end


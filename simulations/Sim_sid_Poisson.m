%% define true parameter values
% currently, the coefficients are set to B_s (identifiable values)
% see below commented part for B_g (generically identifiable values)

K2 = 2;
K1 = 3*K2; J = 3*K1;
B2_sub = zeros(K1, K2); B1_sub = zeros(J, K1);

max_val = 4;
for k = 1:K2
    if k+floor(K2/2) <= K2
        B2_sub([k+2*K2], k) = max_val;
        B2_sub(k+2*K2+floor(K2/2), k) = -max_val/3;
        B2_sub(k+2*K2, k+floor(K2/2)) = -max_val/3;
        B2_sub([k+2*K2+floor(K2/2)], k+floor(K2/2)) = max_val;
    end
    B2_sub([k k+K2], k) = max_val;
end
B2 = [[-max_val/2*ones(K2,1); -max_val*ones(K2,1); -max_val/2*ones(K2,1)], B2_sub];

for k = 1:K1
    if k+floor(K1/2) <= K1
        B1_sub([k+2*K1], k) = 4;
        B1_sub(k+2*K1+floor(K1/2), k) = -4/3;
        B1_sub(k+2*K1, k+floor(K1/2)) = -4/3;
        B1_sub([k+2*K1+floor(K1/2)], k+floor(K1/2)) = 4;
    end
    B1_sub([k k+K1], k) = 4;
end
B1 = [[-2*ones(K1,1); -4*ones(K1,1); -2*ones(K1,1)], B1_sub];

G1 = (B1_sub ~= 0); G2 = (B2_sub ~= 0);
prop_true = 0.5*ones(1, K2); 

%% B_g (generically identifiable values)
% max_val = 4;
% for k = 1:K2
%     B2_sub(k, k:min(k + ceil(K2/3), K2)) = max_val/3;
%     if k+floor(K2/2) <= K2
%         B2_sub([k+2*K2], k) = max_val;
%         B2_sub(k+2*K2+floor(K2/2), k) = -max_val/3;
%         B2_sub(k+2*K2, k+floor(K2/2)) = -max_val/3;
%         B2_sub([k+2*K2+floor(K2/2)], k+floor(K2/2)) = max_val;
%     end
%     B2_sub([k k+K2], k) = max_val;
% end
% B2_sub((K2+1):2*K2, :) = B2_sub(1:K2, :)';
% B2 = [[-2*ones(K2,1); -4*ones(K2,1); -max_val/2*ones(K2,1)], B2_sub];
% 
% for k = 1:K1
%     B1_sub(k, k:min(k + ceil(K1/3), K1)) = max_val/3;
%     if k+floor(K1/2) <= K1
%         B1_sub([k+2*K1], k) = 4;
%         B1_sub(k+2*K1+floor(K1/2), k) = -4/3;
%         B1_sub(k+2*K1, k+floor(K1/2)) = -4/3;
%         B1_sub([k+2*K1+floor(K1/2)], k+floor(K1/2)) = 4;
%     end
%     B1_sub([k k+K1], k) = 4;
% end
% B1_sub((K1+1):2*K1, :) = B1_sub(1:K1, :)';
% % intercept = -10*(sum(B1_sub,2) >= 8) - 5*(sum(B1_sub,2) < 8); % K2 = 6
% intercept = -4*(sum(B1_sub,2) > 5) - 2*(sum(B1_sub,2) <= 5); % K2 = 2
% 
% B1 = [intercept, B1_sub];
% G1 = (B1_sub ~= 0); G2 = (B2_sub ~= 0);
% prop_true = 0.5*ones(1, K2); 



%% main simulation
C = 100; n_vec = [500 1000 2000 4000 8000]; n_parallel = 6;
res_G = zeros(5, 1); res_itera = zeros(5, 1);
res_T = zeros(5, 1); res_time = zeros(5, 1);
K1_max = K1;

for aa = 1:5
    N = n_vec(aa);
    prop_ini = zeros(K2, C);
    prop_final = zeros(K2, C);
    
    tol = K2/2; epsilon = 0.0001;
    lambda_1 = N^(2/8); lambda_2 = N^(2/8); tau = 3*N^(-0.3);
    
    B1_final = zeros(J, K1_max+1, C);
    B1_ini = B1_final;
    B2_final = zeros(K1_max, K2+1, C);
    B2_ini = B2_final;
    
    loglik_est = zeros(C, 1);
    
    err_G1_ini = zeros(C,1);
    err_G1_final = zeros(C,1);
    err_G2_ini = zeros(C,1);
    err_G2_final = zeros(C,1);
    time_vec = zeros(C,1);
    itera = zeros(C,1);
    
    parfor(c= 1:C, n_parallel)
        rng(50+c);
        [X, A1_true, A2_true] = generate_X_poisson(N, prop_true, B1, B2);
        
        tic;
        % spectral initialization
        [prop_ini(:,c), B1_ini(:,:,c), B2_ini(:,:,c), A1_est, A2_est, err_G1_ini(:,c), err_G2_ini(:,c)] = Poi_init(X, K1_max, K1, K2, B1, B2, G1, G2, epsilon);
        
        % SAEM
        [prop_final(:,c), B1_final(:,:,c), B2_final(:,:,c), A1_final, A2_final, itera(c)] = get_SAEM_poisson( ...
             X, prop_ini(:,c)', B1_ini(:,:,c), B2_ini(:,:,c), lambda_1, ...
             lambda_2, tau, A1_est, A2_est, 1, tol);

        % uncomment below to implement PEM instead of SAEM
%         [prop_final(:,c), B1_final(:,:,c), B2_final(:,:,c), ~, loglik_est(c)] = get_EM_poisson( ...
%              X, prop_ini(:,c)', B1_ini(:,:,c), B2_ini(:,:,c), lambda_1, lambda_2, tau, tol);
    
        time_vec(c) = toc;
        fprintf('%d-th iteration complete \n', c);
    end    
    
    ind = (1:C); 
    mse_prop = 0; mse_B1 = 0; mse_B2 = 0; mse_theta = zeros(C,1);
    err_G1 = zeros(C,1); err_G2 = zeros(C,1); err_G_final = zeros(C,1); 
    
    for c = ind
        B1_est = thres(B1_final(:, :, c), tau);
        B2_est = thres(B2_final(:, :, c), tau);
        err_G_final(c) = (sum(G1 == (B1_est(:,2:end) ~= 0), 'all')+sum(G2 == (B2_est(:,2:end) ~= 0), 'all'))/(J*K1+K1*K2);
        err_G1_final(c) = mean(G1 == (B1_est(:,2:end) ~= 0), 'all');
        err_G2_final(c) = mean(G2 == (B2_est(:,2:end) ~= 0), 'all');
        
        mse_theta(c) = (sum((prop_final(:,c)' - prop_true).^2) + sum((B1_est - B1).^2, 'all') + ...
            sum((B2_est - B2).^2, 'all'))/(K2+ K1*(K2+1) + J*(K1+1));
        mse_prop = mse_prop + mean((prop_final(:,c)' - prop_true).^2);
        mse_B1 = mse_B1 + mean((B1_est - B1).^2, 'all');
        mse_B2 = mse_B2 + mean((B2_est - B2).^2, 'all');
    end
    
    % compute final values reported in figures
    res_G(aa) = mean(err_G_final);
    res_T(aa) = sqrt(mean(rmoutliers(mse_theta)));
    res_time(aa) = mean(time_vec);
    res_itera(aa) = mean(itera)-1;
    save("Poisson_sid_SAEM_"+N+"_"+K2+'.mat')
end



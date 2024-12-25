function [prop_ini, B1_ini, B2_ini, A1_est, A2_est] = Poi_init_data(X, K1, K2, epsilon)
    % This code is a modification of the 'Poi_init' (double-SVD initialization) 
    % function by removing column permutation adjustments, and is used to 
    % analyze the 20 newsgroups dataset
    
    %% 1st layer
    [N,J] = size(X);
    [U,S,V] = svd(X, "econ");
    K1_max = K1;
    
    m = max(K1_max+1, sum(diag(S) > 1.01*sqrt(N)));
    
    X_top_m = U(1:end, 1:m) * S(1:m, 1:m) * V(1:end, 1:m)';
    X_trunc = max(X_top_m, epsilon);
    X_inv = log(X_trunc);
    X_inv_adj = X_inv - ones(N,1) * mean(X_inv);
    
    [~, ~, V_true_adj] = svd(X_inv_adj, "econ");

    [R_V, ~] = rotatefactors(V_true_adj(:,1:K1_max), 'method', 'varimax', 'Maxit', 10000);
    B1_est = thres(R_V, 1/2.5/sqrt(J));

    Sign_flip_multiplier_B = [repmat(2*(mean(B1_est(:,1:end), 1) > 0)-1, J, 1)];
    B1_est = B1_est .* Sign_flip_multiplier_B;
   
    A1_est_nonbin = X_inv_adj * B1_est * inv(B1_est' * B1_est);
    A1_est = double(A1_est_nonbin > 0);
    G1_est = (B1_est ~= 0);
    
    A1_centered = A1_est - ones(N,1) * mean(A1_est, 1);
    B1_re_est = (inv(A1_centered' * A1_centered) * A1_centered' * X_inv_adj)';
    B1_re_est = B1_re_est .* G1_est;
    b1 = mean(X_inv_adj, 1)' - B1_re_est * mean(A1_est, 1)' -2;
    B1_ini = [b1, B1_re_est];

    %% 2nd layer
    [U,S,V] = svd(A1_est, "econ");
    m = max(K2+1, sum(diag(S) > 1.01*sqrt(N)));
    A1_top_m = U(1:end, 1:m) * S(1:m, 1:m) * V(1:end, 1:m)';
    A1_trunc = max(min(A1_top_m, 1-epsilon), epsilon);

    A_inv = logit(A1_trunc);
    A_inv_adj = A_inv - ones(N,1)*mean(A_inv);
    
    [~, ~, V_adj] = svd(A_inv_adj, "econ"); 
    [R_V, ~] = rotatefactors(V_adj(:,1:K2), 'method', 'varimax', 'Maxit', 10000);
    B2_est = thres(R_V, 1/2.5/sqrt(K1));
    
    Sign_flip_multiplier_B = [repmat(2*(mean(B2_est(:,1:end), 1) > 0)-1, K1_max, 1)];
    B2_est = B2_est .* Sign_flip_multiplier_B;

    G2_est = double(B2_est ~=0);

    A2_est = A_inv_adj * B2_est * inv(B2_est' * B2_est);
    A2_est = double(A2_est > 0);
    A2_centered = A2_est - ones(N,1) * mean(A2_est, 1);
    B2_re_est = (inv(A2_centered' * A2_centered) * A2_centered' * A_inv_adj)';
    B2_re_est = B2_re_est .* G2_est;
    b2 = mean(A_inv_adj, 1)' - B2_re_est * mean(A2_est, 1)' -2;
    B2_ini = [b2, B2_re_est];
    prop_ini = mean(A2_est > 0);
end

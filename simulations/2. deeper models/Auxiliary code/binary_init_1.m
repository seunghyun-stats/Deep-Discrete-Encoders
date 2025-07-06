function A2_est = binary_init_1(N, K1, K2, A1_est, factor, epsilon)
% given a binary matrix A1_est (N x K1), estimate the upper-layer latent variable
% matrix A2_est (N x K2)

    [U,S,V] = svd(A1_est, "econ");
    m = max(min(K2*2, K1-2), sum(diag(S) > 1.01*sqrt(N)));
    A1_top_m = U(1:end, 1:m) * S(1:m, 1:m) * V(1:end, 1:m)';
    A1_trunc = max(min(A1_top_m, 1-epsilon), epsilon);

    A_inv = logit(A1_trunc);
    A_inv_adj = A_inv - ones(N,1)*mean(A_inv);
    [~, ~, V_adj] = svd(A_inv_adj, 'econ'); 

    % trivial case with only one latent variable
    if K2 == 1
        B2_est = V_adj(:,1);
        Sign_flip_multiplier_B = [repmat(2*(mean(B2_est(:,1:end), 1) > 0)-1, K1, 1)];
        B2_est = B2_est .* Sign_flip_multiplier_B;

        A2_est = A_inv_adj * B2_est * inv(B2_est' * B2_est);
        A2_est = double(A2_est > 0);
        A2_centered = A2_est - ones(N,1) * mean(A2_est, 1);
        B2_re_est = (inv(A2_centered' * A2_centered) * A2_centered' * A_inv_adj)'/factor;

    else
        % varimax
        [R_V, ~] = rotatefactors(V_adj(:,1:K2), 'method', 'varimax', 'Maxit', 10000);
        B2_est = thres(R_V, max(0.05, 1/2.5/sqrt(K1)));
        
        % adjust sign flip
        Sign_flip_multiplier_B = [repmat(2*(mean(B2_est(:,1:end), 1) > 0)-1, K1, 1)];
        B2_est = B2_est .* Sign_flip_multiplier_B;

        % estimate A2
        A2_est = A_inv_adj * B2_est * inv(B2_est' * B2_est);
        A2_est = double(A2_est > 0);
    end
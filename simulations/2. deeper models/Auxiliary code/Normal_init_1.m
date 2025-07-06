function A1_est = Normal_init_1(X, K1, B1)
% given a continuous matrix X (N x J), estimate the upper-layer binary 
% latent variable matrix A1_est (N x K1)

    [N,J] = size(X);
    X_centered = X- ones(N,1) * mean(X);
    [~, ~, V] = svd(X_centered, "econ");

    %% 1st layer
    m = K1;
    V_top_m = V(:,1:m);

    [R_V, ~] = rotatefactors(V_top_m(:,1:K1), 'method', 'varimax', 'Maxit', 10000);
    B1_est = thres(R_V, 1/2.5/sqrt(J));

    % adjust sign flip
    Sign_flip_multiplier_B = [repmat(2*(mean(B1_est(:,1:end), 1) > 0)-1, J, 1)];
    B1_est = B1_est .* Sign_flip_multiplier_B;
    
    % adjust column permutation
    costmat = zeros(K1, K1);
    for k = 1:K1
        costmat(k,:) = - sum(B1_est(find(B1(:,k+1) > 0), :), 1);
    end
    [assignment, ~] = munkres(costmat);
    B1_est = B1_est(:, assignment);

    % estimate A1
    A1_est_nonbin = X_centered * B1_est * inv(B1_est' * B1_est);
    A1_est = double(A1_est_nonbin > 0);
end

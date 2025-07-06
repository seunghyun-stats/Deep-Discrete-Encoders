function [X, A] = generate_X_normal_D(N, prop_true, D, B_cell, gamma)
% Generate data from a Normal-D-latent-layer DDE
% @param N: sample size
% @param prop_true: length K{D} vector for the top-latent-layer proportions
% @param B_cell: length D cell conatining each layers' coefficients
% @param gamma: legnth J vector of the bottom layer variances

    K_top = size(prop_true, 2);
    A = cell(D,1);

    % generate A{D}
    A_top = zeros(N, K_top);
    for k = 1:K_top
        A_top(:,k) = binornd(1, prop_true(k), N, 1);
    end
    A{D} = A_top;

    % recursively generate A{d}
    for d = (D-1):-1:1
        A{d} = binornd(1, logistic([ones(N,1), A{d+1}] * B_cell{d+1}'));
    end

    % generate the observed data X
    X = normrnd([ones(N,1), A{1}] * B_cell{1}', repmat(sqrt(gamma'),N,1));
end
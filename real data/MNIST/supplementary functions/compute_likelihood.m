function [loglik] = compute_likelihood(X, prop, B1, B2)
    K2 = size(prop, 2);
    [J, K1] = size(B1);
    K1 = K1-1;
    
    N = size(X, 1);
    
    A2 = binary(0:(2^K2-1), K2);
    A1 = binary(0:(2^K1-1), K1);
    phi = zeros(N, 2^K1, 2^K2); % indexed by (i, a, b)
    
    for a = 1:2^K1
        lambda = [1, A1(a,:)] * B1'; % A1_long(a,:) * B1'
        for b = 1:2^K2
            eta = [1, A2(b,:)] * B2';
            phi(:, a, b) = exp(sum(X.*lambda, 2) - sum(log(1+exp(lambda))) + sum(A1(a,:).*eta) ...
                - sum(log(1+exp(eta))) + log(prop) * A2(b,:)' + log(1 - prop) * (1-A2(b,:))');
        end
    end
    loglik = sum(log(sum(sum(phi(:, :, :), 3), 2)));
end

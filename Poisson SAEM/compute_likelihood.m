function [loglik] = compute_likelihood(X, prop, B1, B2)
    K2 = size(prop, 2);
    [J, K1] = size(B1);
    K1 = K1-1;
    
    N = size(X, 1);
    
    A2 = binary(0:(2^K2-1), K2);
    A2_long = [ones(2^K2,1), A2];
    A1 = binary(0:(2^K1-1), K1);
    A1_long = [ones(2^K1,1), A1];
    phi = zeros(N, 2^K1, 2^K2); % indexed by (i, a, b)
    phi_2 = zeros(N,1);

    for i = 1:N
        for a = 1:2^K1
            lambda = exp(A1_long(a,:) * B1'); % A1_long(a,:) * B1'
            for b = 1:2^K2
                eta = A2_long(b,:) * B2';
                phi(i, a, b) = exp(-sum(lambda) + X(i,:)*log(lambda)' + sum(A1(a,:).*eta) ...
                    - sum(log(1+exp(eta))) + log(prop) * A2(b,:)' + log(1 - prop) * (1-A2(b,:))');
            end
        end
        phi_2(i) = sum(phi(i, :, :), 'all');
    end

    loglik = sum(log(phi_2)) - sum(log(factorial(X)), 'all');
end


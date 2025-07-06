function [pi_1, pi_2] = posterior_mean(phi,K1,K2)
% input: phi (as a N x 2^(K1+K2) matrix), K1, K2
% output: pi_1 (posterior mean of A1, size N x K1), pi_2 (posterior mean of A2, size N x K2)

    N = size(phi,1);
    phi_re = reshape(phi, N, 2^K1, 2^K2);

    pi_1 = sum(phi_re, 3); A1 = binary(0:(2^K1-1), K1);
    pi_1 = pi_1*A1;

    pi_2 = reshape(sum(phi_re, 2), N, 2^K2); A2 = binary(0:(2^K2-1), K2);
    pi_2 = pi_2*A2;
end


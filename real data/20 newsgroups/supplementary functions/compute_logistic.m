function [scaled, log_sum] = compute_logistic(M)
% M is 2^K1 x 2^K2 matrix
    max_M = max(M,[], "all");
    M_centered = M - max_M; ind = M_centered > -8; M_centered(logical(1-ind)) = -Inf;
    log_sum = log(sum(exp(M_centered(ind))));
    scaled = zeros(size(M)); scaled(ind) = exp(M_centered(ind) - log_sum);
    log_sum = log_sum + max_M;
end

function [f] = F_1_SAEM(Xj, A1_sample_long, N, K1, C) % for fixed j
% return negative loglik
% currently, pen is a scalar

f = @loglike;
    function [ll] = loglike(x)
        eta = reshape(sum(bsxfun(@times, A1_sample_long, reshape(x, [1 K1 + 1 1])), 2), [N C]);
        ll = sum((Xj - eta).^2, 'all')/C;
        % -(sum(eta,2)' * Xj - sum( log(exp(eta)+1), 'all'))/C;
    end
end
function [f] = Fun_1_TLP_EM(X_j, A1_long, pi, pen, tau)  
% return penalized loglik
% currently, pen is a scalar

f = @loglike; % NEW
    function [ll] = loglike(x) 
        penalty = pen * TLP(x, tau);
        eta = x * A1_long';
        obj = sum((X_j - eta).^2 .* pi, 'all'); % - exp(eta)*pi_0 + x * [ones(size(X_j,1),1), pi_1]' * X_j;
        ll = penalty + obj;
    end
end
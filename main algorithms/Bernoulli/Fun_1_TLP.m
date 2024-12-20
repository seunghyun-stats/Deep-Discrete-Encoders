function [f] = Fun_1_TLP(X_j, A1_long, pi_1, pi_0, pen, tau) 
% return penalized loglik
% currently, pen is a scalar

f = @loglike;
    function [ll] = loglike(x) 
        penalty = pen * TLP(x, tau);
        
        eta = x * A1_long';
        obj = x * [ones(size(X_j,1),1), pi_1]' * X_j - log(exp(eta)+1)*pi_0;
        ll = penalty - obj;

    end
end
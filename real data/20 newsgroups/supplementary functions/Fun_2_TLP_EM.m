function [f] = Fun_2_TLP_EM(A2_long, psi_k, psi_2, pen, tau) 
% return penalized loglik

f = @loglike;
    function [ll] = loglike(x) 
        penalty = pen * TLP(x, tau);

        eta = x * A2_long';
        obj = eta * psi_k' - log(exp(eta)+1) * psi_2;
        ll = penalty - obj;
    end
end
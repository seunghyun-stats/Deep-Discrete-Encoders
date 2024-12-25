function [f] = Fun_2_confirm(A2_long, index, psi_k, psi_2, K2) 

f = @loglike;
    function [ll] = loglike(x)
        y = zeros(1, K2+1); y(index) = x;
        eta = y * A2_long';
        obj = eta * psi_k' - log(exp(eta)+1) * psi_2;
        ll = - obj;
    end
end
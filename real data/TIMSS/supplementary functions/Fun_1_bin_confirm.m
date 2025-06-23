function [f] = Fun_1_bin_confirm(R_j, A1_j, psi, pi_0)

f = @loglike;
    function [obj] = loglike(x) 
        eta_j = x * A1_j';
        obj = - eta_j*psi'*R_j + log(exp(eta_j)+1)*pi_0;
    end
end

function [f] = Fun_1_confirm(X_j, A1_j, psi)  

f = @loglike;
    function [obj] = loglike(x) 
        eta_j = x * A1_j';
        obj = sum(psi .* (X_j- eta_j).^2, 'all');
    end
end
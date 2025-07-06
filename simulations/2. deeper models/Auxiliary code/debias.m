function X_debias = debias(X, tau)
    index_1 = X>tau; index_2 = X<tau; X_debias = X;
    X_debias(index_1) = X(index_1) + tau;
    X_debias(index_2) = X(index_2) - tau;
    index = abs(X) < tau;
    X_debias(index) = 0;
end
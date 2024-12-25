function [prop, B1, B2, gamma, loglik, itera] = get_EM_normal_confirm(X, prop_in, B1_in, B2_in, gamma_in, G1, G2)

% confirmatory EM algorithm with fixed G1 and G2 for the normal-2-layer DDE
% this code was used for analyzing the TIMSS response time dataset

K2 = size(prop_in,2);
[J, K1] = size(G1);
N = size(X, 1);

A2 = binary(0:(2^K2-1), K2);
A2_long = [ones(2^K2,1), A2];
A1 = binary(0:(2^K1-1), K1);
A1_long = [ones(2^K1,1), A1];

prop = prop_in; prop_update = prop;
B1 = B1_in; B1_update = B1;
B2 = B2_in; B2_update = B2;
gamma = gamma_in; gamma_update = gamma;

S1 = sum(G1, 2);
index1 = cell(J,1);
for j = 1:J
    index1{j} = [1, 1+find(G1(j,:))];
end

S2 = sum(G2, 2);
index2 = cell(K1,1);
for k = 1:K1
    index2{k} = [1, 1+find(G2(k,:))];
end

err = 1;
itera = 0;
loglik = 0;

iter_indicator = (abs(err) > 5*1e-2 && itera < 1000);

% optimization constraints (lower, upper bound) for [beta(j,:), gamma(j)]
lb_1 = cell(J,1); ub_1 = cell(J,1); 
lb_2 = cell(K1,1); ub_2 = cell(K1,1);

for j = 1:J
    lb_1{j} = [-2, zeros(1, S1(j))];
    ub_1{j} = 6*ones(1, S1(j)+1);
end
for k = 1:K1
    lb_2{k} = [-4, zeros(1, S2(k))];
    ub_2{k} = 6*ones(1, S2(k)+1);
end

bb = []; Aeq = []; beq = [];
AA = []; %lb = []; ub = [];

options = optimset('Display', 'off'); 

while iter_indicator
    %% E-step
    old_loglik = loglik; loglik_prev_step = -N/2*sum(log(2*pi*gamma)) + N*sum(log(1-prop));
    
    phi = zeros(N, 2^K1, 2^K2);
    
    eta_1 = B1*A1_long';
    eta_2 = B2*A2_long';
    
    for i = 1:N
        exponent = zeros(2^K1, 2^K2);
        for a = 1:2^K1
            for b = 1:2^K2
                exponent(a,b) = -sum((X(i,:)'- eta_1(:,a)).^2./gamma)/2 + A1(a,:) * eta_2(:,b) - sum(log(1+exp(eta_2(:,b)))) ...
                                + log(prop ./ (1-prop)) * A2(b,:)';
            end
        end
        phi(i, :, :) = exp(exponent - max(max(exponent))); % has NaNs??
        tmp = sum(phi(i, :, :), 'all');
        phi(i,:,:) = phi(i,:,:)/tmp;
        loglik_prev_step = loglik_prev_step + log(tmp) + max(max(exponent)) ;
    end
    psi = sum(phi, 3);
    psi_2 = zeros(2^K2,1); psi_3 = zeros(K1, 2^K2); 
    for b = 1:2^K2
        psi_2(b) = sum(phi(:,:,b), "all");
    end

    for k = 1:K1
        for b = 1:2^K2
            psi_3(k, b) = sum(phi(:, :, b),1) * A1(:,k);
        end
    end

    %% M-step
    % prop
    for k = 1:K2
        prop_update(k) = psi_2' * A2(:,k) /N;
    end
    
    %% B1
    for j = 1:J
        f_j = Fun_1_confirm(X(:,j), A1_long(:,index1{j}), psi);
        opt = fmincon(f_j, nonzeros(B1(j,:))', AA, bb, Aeq, beq, lb_1{j}, ub_1{j}, nonlcon, options);
        B1_update(j, index1{j}) = opt;
        gamma_update(j) = f_j(opt)/N;
    end

    %% B2 - check the code...
    for k = 1:K1
        f2 = Fun_2_confirm(A2_long, index2{k}, psi_3(k,:), psi_2, K2);
        opt_2 = fmincon(f2, nonzeros(B2(k,:))', AA, bb, Aeq, beq, lb_2{k}, ub_2{k}, nonlcon, options);
        B2_update(k,index2{k}) = opt_2;
    end

    %% updating log-lik
    loglik = loglik_prev_step;
    err = (loglik - old_loglik);
    iter_indicator = (abs(err) > 5* 1e-2 && itera<1000);
    itera = itera + 1;
    fprintf('EM Iteration %d,\t Err %1.8f\n', itera, err);

    prop = prop_update; B1 = B1_update; B2 = B2_update; gamma = gamma_update;
end

end
function [prop, B1, B2, phi, loglik] = get_EM_bernoulli(X, prop_in, B1_in, B2_in, pen_1, pen_2, tau, tol)
%% Algorithm 6 (Basic PEM) for Bernoulli-two-latent-layer DDEs
% @X: N x J binary data matrix
% @prop_in, B1_in, B2_in: initialization for corresponding parameters
% @pen_1, pen_2, tau: tuning parameters for the TLP penalty; pen_1, pen_2 corresponds to the magnitude (lambda in the paper)
%     and tau denotes the threshold valu
% @tol: tolerance for convergence


% definitions
K2 = size(prop_in,2);
[J, K1] = size(B1_in);
K1 = K1-1;

N = size(X, 1);

A2 = binary(0:(2^K2-1), K2);
A2_long = [ones(2^K2,1), A2];
A1 = binary(0:(2^K1-1), K1);
A1_long = [ones(2^K1,1), A1];

prop = prop_in;
B1 = B1_in;
B2 = B2_in;

% iteration settings
t = 1;
iter_indicator = true;
loglik = 0;

prop_update = prop_in;
B1_update = zeros(J, K1+1);
B2_update = zeros(K1, K2+1);

% optimization settings
bb = []; Aeq = []; beq = [];
AA = [];
lb_1 = [-10*ones(1,1), -10*ones(1, K1)];
ub_1 = 10*ones(1, K1+1);

lb_2 = [-10*ones(1,1), -10*ones(1, K2)];
ub_2 = 10*ones(1, K2+1);
options = optimset('Display', 'off', 'MaxIter', 20); 

phi = zeros(N, 2^K1, 2^K2);
psi = zeros(K1, 2^K2);
psi_2 = zeros(2^K2, 1);
xi = zeros(1, 2^K2);

%%% iteration start
while iter_indicator
    old_loglik = loglik;

    %% E-step
    for a = 1:2^K1
        lambda = [1, A1(a,:)] * B1';
        for b = 1:2^K2
            eta = [1, A2(b,:)] * B2';
            phi(:, a, b) = exp(sum(X.*lambda, 2) - sum(log(1+exp(lambda))) + sum(A1(a,:).*eta) ...
                - sum(log(1+exp(eta))) + log(prop) * A2(b,:)' + log(1 - prop) * (1-A2(b,:))');
        end
    end
    loglik = sum(log(sum(sum(phi(:, :, :), 3), 2)));

    phi_2 = sum(sum(phi, 3), 2);
    phi = phi ./ repmat(phi_2, 1, 2^K1, 2^K2);

    for b = 1:2^K2
        xi(b) = sum(phi(:, :, b), 'all');
    end

    for a0 = 1:K1
        for b = 1:2^K2
            psi(a0, b) = sum(phi(:, :, b),1) * A1(:,a0);
        end
    end

    for b = 1:2^K2
        psi_2(b) = sum(phi(:,:,b), "all");
    end

    pi = sum(phi, 3);
    pi_0 = sum(pi, 1)';
    pi_1 = pi*A1;
    
    %% M-step
    % prop
    for k = 1:K2
        prop_update(k) = xi * A2(:,k) /N;
    end

    % B1
    for j = 1:J
        f_j = Fun_1_TLP(X(:,j), A1_long, pi_1, pi_0, pen_1, tau);
        B1_update(j,:) = fmincon(f_j, B1(j,:), AA, bb, Aeq, beq, lb_1, ub_1, nonlcon, options);
    end

    % B2 
    for k = 1:K1
        f2 = Fun_2_TLP(A2_long, psi(k,:), psi_2, pen_2, tau);
        B2_update(k,:) = fmincon(f2, B2(k,:), AA, bb, Aeq, beq, lb_2, ub_2, nonlcon, options);
    end
    

    %% compute error
    err = loglik - old_loglik;
    prop = prop_update;
    B1 = [B1_update(:, 1), thres(B1_update(:,2:end), tau)];
    B2 = [B2_update(:, 1), thres(B2_update(:,2:end), tau)];
    t = t + 1;
    
    fprintf('EM Iteration %d,\t Err %1.1f \n', t, err);
    iter_indicator = ( abs(err) > tol & t < 50);
end

end

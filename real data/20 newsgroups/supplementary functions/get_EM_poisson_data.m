function [prop, B1, B2, phi, loglik, itera] = get_EM_poisson_data(X, prop_in, B1_in, B2_in, pen_1, pen_2, tau, tol)
% This function is a modification of the function 'get_EM_poisson' (PEM algorithm)
% by using the log-sum-exp trick to deal with the many zeros in the
% dataset.

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
itera = 0;
iter_indicator = true;
loglik = 0;

prop_update = zeros(1,K2);
bb = []; Aeq = []; beq = [];
AA = []; %lb = []; ub = [];
lb_1 = [-10*ones(1,1), -10*ones(1, K1)]; % zeros(J, K1+1);
ub_1 = 15*ones(1, K1+1);

lb_2 = [-10*ones(1,1), -5*ones(1, K2)]; % zeros(J, K1+1);
ub_2 = 10*ones(1, K2+1);
options = optimset('Display', 'off', 'MaxIter', 15); 

%%% iteration start
while iter_indicator
    old_loglik = loglik;

    %% E-step
    phi = zeros(N, 2^K1, 2^K2);
    phi_2 = zeros(N,1);
    psi = zeros(K1, 2^K2);
    psi_2 = zeros(2^K2, 1);
    xi = zeros(1, 2^K2);

    lambda = A1_long * B1';
    for i = 1:N
        for b = 1:2^K2
            eta = A2_long(b,:) * B2';
            phi(i, :, b) = (-sum(exp(lambda),2 ) + lambda*X(i,:)' + A1*eta' ...
                - sum(log(1+exp(eta))) + log(prop) * A2(b,:)' + log(1 - prop) * (1-A2(b,:))');
        end
        [phi(i,:,:), phi_2(i)] = compute_logistic(reshape(phi(i,:,:),2^K1,2^K2));
    end
    loglik = sum(phi_2); % - partition_ftn;

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
        prop_update(k) = sum(xi, 1) * A2(:,k) /N;
    end
    
    %% B1
    for j = 1:J
        f_j = Fun_1_TLP_EM(X(:,j), A1_long, pi_1, pi_0, pen_1, tau);
        B1_update(j,:) = fmincon(f_j, B1(j,:), AA, bb, Aeq, beq, lb_1, ub_1, nonlcon, options);
    end

    %% B2 
    for k = 1:K1
        f2 = Fun_2_TLP_EM(A2_long, psi(k,:), psi_2, pen_2, tau);
        B2_update(k,:) = fmincon(f2, B2(k,:), AA, bb, Aeq, beq, lb_2, ub_2, nonlcon, options);
    end
     
    %% compute log-lik: actually this is loglik before updating :P
    % loglik = sum(log(phi_2)) - partition_ftn;
    err = sqrt(norm(prop - prop_update, "fro")^2 + norm(B1 - B1_update, "fro")^2 + norm(B2 - B2_update, "fro")^2);
    % err = loglik - old_loglik;

    prop = prop_update;
    B1 = B1_update; % [B1_update(:,1), thres(B1_update(:, 2:end), tau)];
    B2 = B2_update; % [B2_update(:,1), thres(B2_update(:, 2:end), tau)];

    itera = itera + 1;
    fprintf('EM Iteration %d,\t Err %1.1f \n', itera, err);
    iter_indicator = ( abs(err) > tol & itera < 100);
end
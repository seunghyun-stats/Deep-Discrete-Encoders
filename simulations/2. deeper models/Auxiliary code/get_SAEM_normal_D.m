function [prop, B_cell, gamma, A_cell_new, t] = get_SAEM_normal_D(X, D, K_cell, prop_in, B_cell_in, gamma_in, pen_vec, A_cell_in, C, n_par, tol)
% `tol` denotes the convergence criteria; can modify to a larger value for
% faster implementation

%% definitions
prop = prop_in;
pen = pen_vec(1); tau = pen_vec(2);

[J, K1] = size(B_cell_in{1}); K1 = K1-1;

N = size(X, 1);
B_cell = B_cell_in; A_cell_new = A_cell_in;
gamma = gamma_in; 

% iteration settings
t = 1; % iteration index
iter_indicator = true;


% optimization settings
bb = []; Aeq = []; beq = [];
AA = [];
lb_1 = [-5*ones(1,1), -5*ones(1, K1)];
ub_1 = 10*ones(1, K1+1);
options = optimset('Display', 'off', 'MaxIter', 6);

A_sample_long = cell(D,1);

ber = cell(D,1);
for d = 1:D
    ber{d} = zeros(N, K_cell{d});
end

p = zeros(N, K_cell{D});

% optimization functions
ftn_pen = @(x) pen * TLP(x(2:end), tau);


f_old = cell(J,D);
for j = 1:J
    for d = 1:D
        f_old{j,d} = @(x) 0; % for d \ge 2, the later indices are not used.
    end
end

B_cell_update = B_cell;
gamma_update = zeros(J,1);


%% main iteration
while iter_indicator
    A_cell_old = A_cell_new;

    %% Simulation step
    B_D = B_cell{D};
    for i = 1:N
        A_d_i = A_cell_old{D}(i,:);
        for l = 1:K_cell{D}
            A_d_i(l) = 1; eta_1 = B_D * [1, A_d_i]';
            A_d_i(l) = 0; eta_0 = B_D * [1, A_d_i]';
            ber{D}(i,l) = logistic(log(prop(l) / (1 - prop(l))) + ...
                 sum(A_cell_old{D-1}(i,:)' .* B_D(:,l+1)) - sum(log( (1 + exp(eta_1)) ./ (1 + exp(eta_0)))));
        end
    end
    
    for d = 2:(D-1)
        B2 = B_cell{d};
        for i = 1:N
            A_d_i = A_cell_old{d}(i,:);
            for l = 1:K_cell{d}
                A_d_i(l) = 1; eta_1 = B2 * [1, A_d_i]';
                A_d_i(l) = 0; eta_0 = B2 * [1, A_d_i]';
                ber{d}(i,l) = logistic([1, A_cell_old{d+1}(i,:)] * B_cell{d+1}(l,:)' + ...
                    sum(A_cell_old{d-1}(i,:)' .* B2(:,l+1)) - sum(log( (1 + exp(eta_1)) ./ (1 + exp(eta_0)))));
            end
        end
    end

    B1 = B_cell{1};
    for i = 1:N
        A1_i = A_cell_old{1}(i,:);
        for k = 1:K1                              
            ber{1}(i,k) = logistic( [1, A_cell_old{2}(i,:)] * B_cell{2}(k,:)' - sum(( (B1(:,k+1).^2 -2*(X(i,:)' - B1(:, 1:end~=(k+1)) * [1; A1_i(1:end~=k)']) ...
                .*B1(:,k+1) ) )./gamma) /2 );
        end
    end
    
    % sample each latent variable
    for d = 1:D
        A_sample_long{d} = zeros(N, K_cell{d}+1, C);
        for c = 1:C
            A_sample_long{d}(:,:,c) = [ones(N,1), double(rand(N,K_cell{d}) < ber{d})];
        end
    end
    
    %% Stochastic approximation M-step
    % prop
    p = (1-1/t)*p + 1/t*ber{D};
    prop_update = sum(p, 1)/N;

    % B1, gamma
    invvv = zeros(K1+1, K1+1);
    for c = 1:C
        invvv = invvv + reshape(A_sample_long{1}(:,:,c), [N K1+1])'*reshape(A_sample_long{1}(:,:,c), N, K1+1);
    end
    invvv = inv(invvv/C);

    sum_X = zeros(K1+1, J);
    for c = 1:C
        sum_X = sum_X + reshape(A_sample_long{1}(:,:,c), [N K1+1])'*X;
    end
    sum_X = sum_X/C;
    
    B1_update = (invvv * sum_X)';
    B1_update(:,2:end) = thres(B1_update(:,2:end), tau);
    for j =1:J
        Xj = X(:,j);
        eta = reshape(sum(bsxfun(@times, A_sample_long{1}, reshape(B1_update(j,:), [1 K1 + 1 1])), 2), [N C]);
        gamma_update(j) = sum((Xj - eta).^2, 'all')/(N*C);
    end
    B_cell_update{1} = B1_update;

    % B_2~B_D
    for d = 2:D
        for k = 1:K_cell{d-1}
            K_d = K_cell{d};
            A_sample_k = reshape(A_sample_long{d-1}(:,k+1,:), [N C]);
            f_loglik = F_2_SAEM(A_sample_k, A_sample_long{d}, N, K_d, C);
            f_old{k,d} = @(x) (1-1/t) * f_old{k,d}(x) + 1/t * f_loglik(x);
            f_k = @(x) f_old{k,d}(x) + ftn_pen(x);
            B_cell_update{d}(k,:) = fmincon(f_k, B_cell{d}(k,:), AA, bb, Aeq, beq,  [-5*ones(1,1), -5*ones(1, K_d)], ...
                10*ones(1, K_d+1), nonlcon, options);
        end
    end
    

    %% compute error
    err = 0;
    for d = 1:D
        B_cell_update{d} = [B_cell_update{d}(:, 1), debias(B_cell_update{d}(:,2:end), tau)];
        err = err +  norm(B_cell_update{d}.*(B_cell{d} ~=0) - B_cell{d}, "fro")^2;
        A_cell_new{d} = double(mean(A_sample_long{d}(:,2:end,:), 3) > 0.5); 
    end
    B_cell = B_cell_update;
    prop = prop_update;
    gamma = gamma_update;
    err = sqrt((err + norm(prop - prop_update, "fro")^2 + norm(gamma - gamma_update, "fro")^2)/(n_par));
    
    fprintf('EM Iteration %d,\t Err %1.2f\n', t, err);
    t = t + 1;

    iter_indicator = ( abs(err) > tol & t < 6); % max iterations
end

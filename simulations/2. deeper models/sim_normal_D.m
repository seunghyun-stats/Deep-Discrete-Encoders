% This code was used to generate Table 1 and Tables S.3, S.4 (in the
% supplement).

addpath('C:\Users\leesa\Documents\MATLAB\Deep parametric model\GitHub codes\utilities')
addpath('C:\Users\leesa\Documents\MATLAB\Deep parametric model\GitHub codes\main algorithms\Normal')
addpath('Auxiliary code')


%% Define number of latent layers

% the following script is set to D = 4 latent layers. 
% comment/uncomment the code blocks appropriately to simulate D = 3 or 5.

D = 4; K4 = 2;  % modify to D=3; K3 = 2; or D=5; K5 = 2;


%% define parameters
max_val = 4; 
% K4 = 3*K5; 
K3 = 3*K4; K2 = 3*K3; K1 = 3*K2; J = 3*K1;

% B5 = generate_B(K4,K5,max_val); G5 = B5(:,2:end) ~= 0;
B4 = generate_B(K3,K4,max_val); G4 = B4(:,2:end) ~= 0;
B3 = generate_B(K2,K3,max_val); G3 = B3(:,2:end) ~= 0;
B2 = generate_B(K1,K2,max_val); G2 = B2(:,2:end) ~= 0;
B1 = generate_B(J,K1,max_val); G1 = B1(:,2:end) ~= 0;

B_cell_true = cell(D,1); 
B_cell_true{1} = B1; B_cell_true{2} = B2; B_cell_true{3} = B3; B_cell_true{4} = B4; % B_cell_true{5} = B5; 

K_cell = cell(D,1);
K_cell{1} = K1; K_cell{2} = K2; K_cell{3} = K3; K_cell{4} = K4; % K_cell{5} = K5; 

prop_true = 0.5*ones(1, K4);    % modify to K3 or K5
gamma = ones(J,1);
n_par = K3 + J + 4*(K1+K2+K3+K4);  % remove K4 or add K5


%% main simulation
C = 100; n_vec = [500 1000 2000 4000 8000 16000]; 
n_parallel = 4; 
res_itera = zeros(length(n_vec), 1); res_time = zeros(length(n_vec), 1);
err_Gd_final = zeros(C,D);

for aa = 1:length(n_vec)
    N = n_vec(aa);
    prop_ini = zeros(K4, C);
    prop_final = zeros(K4, C);
    
    tol = K4/2; epsilon = 0.0001;
    pen_vec = 2 * [N.^([3/8]*0.9-1/2), 0.3];
    
    B_cell_fin = cell(C);
    gamma_ini = zeros(J, C);
    gamma_final = zeros(J, C);

    time_vec = zeros(C,1);
    
    parfor(c= 1:C, n_parallel)
        rng(50+c);
        [X, A_cell_true] = generate_X_normal_D(N, prop_true, D, B_cell_true, gamma);
        
        tic;
        % spectral initialization
        [prop_ini(:,c), B_cell_ini, gamma_ini(:,c), A_cell_est] = Normal_init_D(X, D, K_cell, B_cell_true, epsilon);

        % SAEM
        [prop_final(:,c), B_cell_fin{c}, gamma_final(:,c), ~, ~] = get_SAEM_normal_D( ...
             X, D, K_cell, prop_ini(:,c)', B_cell_ini, gamma_ini(:,c), pen_vec, A_cell_est, 1, n_par, 0.3);
        
        time_vec(c) = toc;
        fprintf('%d-th iteration complete \n', c);
    end    
    
    % compute error metrics
    tau = pen_vec(2);
    
    for c = 1:C
        B1_final = B_cell_fin{c}{1}; B2_final = B_cell_fin{c}{2}; B3_final = B_cell_fin{c}{3}; 
        B4_final = B_cell_fin{c}{4}; % B5_final = B_cell_fin{c}{5}; 

        B1_est = [B1_final(:, 1), thres(B1_final(:, 2:end), tau)];
        B2_est = [B2_final(:, 1), thres(B2_final(:, 2:end), tau)];
        B3_est = [B3_final(:, 1), thres(B3_final(:, 2:end), tau)];
        B4_est = [B4_final(:, 1), thres(B4_final(:, 2:end), tau)];
        % B5_est = [B5_final(:, 1), thres(B5_final(:, 2:end), tau)];
        
        err_Gd_final(c,1) = mean(G1 == (B1_est(:,2:end) ~= 0), 'all');
        err_Gd_final(c,2) = mean(G2 == (B2_est(:,2:end) ~= 0), 'all');
        err_Gd_final(c,3) = mean(G3 == (B3_est(:,2:end) ~= 0), 'all');
        err_Gd_final(c,4) = mean(G4 == (B4_est(:,2:end) ~= 0), 'all');
        % err_Gd_final(c,5) = mean(G5 == (B5_est(:,2:end) ~= 0), 'all');
    end

    mean(err_Gd_final)                      % accuracy of each layer graphical matrix
    res_time(aa) = mean(time_vec);          % average computation time
    save("Gaussian_sid_"+D+"_"+N+'.mat')    % final output is stored as a .mat file
end

%% import data
addpath('dataset')
addpath('supplementary functions')

K1 = 7; K2 = 1;

G1 = importdata("Q.csv").data; G2 = ones(K1,1); 
X = importdata("TIMSS.csv");

R = X(:, 1:29);  % response accuracy
X = X(:, 30:58); % response time

N = size(X,1);
J = size(X,2);


%% main estimation
rng(5);

prop_in = 0.4 + 0.4*rand(1,K2);
gamma_in = 0.4*ones(J,1);
B1_time_in = [2*ones(J,1)+rand(J,1), G1.*(rand(J, K1)+0.5)];
B1_resp_in = [-1*ones(J,1)+rand(J,1), G1.*(rand(J, K1))];
B2_in =[-1.5*ones(K1,1)+rand(K1,1), G2.*(2*rand(K1, K2)+0.5)];

[prop, B1_time, B1_resp, B2, gamma, loglik, itera] = get_EM_multimodal_confirm(X, R, prop_in, B1_time_in, B1_resp_in, B2_in, gamma_in, G1, G2);

%% visualize estimated coefficients (Figure S.16 in the supplement)
figure;
h = heatmap(0:4, 1:J, round(B1_time(1:J,1:5),2));
h.Title = 'Response time';
h.XLabel = 'attribue indices';
h.XData = ["intercept","Number","Algebra", 'Geometry', "Data&Prob"];
s = struct(h);
s.XAxis.TickLabelRotation = 30;
print('-r300', 'RT', '-dpng')

figure;
h = heatmap(0:4, 1:J, round(B1_resp(1:J,1:5),2));
h.Title = 'Response accuracy';
h.XLabel = 'attribue indices';
h.XData = ["intercept","Number","Algebra", 'Geometry', "Data&Prob"];
s = struct(h);
s.XAxis.TickLabelRotation = 30;
print('-r300', 'RA', '-dpng')



%% estimate latent configurations
phi = zeros(N, 2^K1, 2^K2);
A1 = binary(0:(2^K1-1), K1); A1_long = [ones(2^K1,1), A1];
A2 = binary(0:(2^K2-1), K2); A2_long = [ones(2^K2,1), A2];
eta_time_1 = B1_time*A1_long'; eta_resp_1 = B1_resp*A1_long';
eta_2 = B2*A2_long';

for i = 1:N
    exponent = zeros(2^K1, 2^K2);
    for a = 1:2^K1
        for b = 1:2^K2
            exponent(a,b) = R(i,:)*eta_resp_1(:,a) - sum(log(1+exp(eta_resp_1(:,a)))) ...
                    -sum((X(i,:)'- eta_time_1(:,a)).^2./gamma)/2 + A1(a,:) * eta_2(:,b) - sum(log(1+exp(eta_2(:,b)))) ...
                    + log(prop ./ (1-prop))' * A2(b,:)';
        end
    end
    phi(i, :, :) = exp(exponent - max(max(exponent)));
    tmp = sum(phi(i, :, :), 'all');
    phi(i,:,:) = phi(i,:,:)/tmp;
end

% A2
psi = zeros(N, 2^K2); A2_est = zeros(N, K2);
for i = 1:N
    for b = 1:2^K2
        psi(i, b) = sum(phi(i,:,b), "all");
    end
end
for i = 1:N
    [~, b] = max(psi(i,:));
    A2_est(i,:) = binary(b-1, K2);
end

% A1
psi_a = zeros(N, 2^K1); A1_est = zeros(N, K1);
for i = 1:N
    for a = 1:2^K1
        psi_a(i, a) = sum(phi(i,a,:), "all");
    end
end
for i = 1:N
    [~, b] = max(psi_a(i,:));
    A1_est(i,:) = binary(b-1, K1);
end


%% comparison with held-out survey response (Table 6 in the main paper)
% Table 6 reports `mean_vec_2` and the first 4 columns of `mean_vec_1`

like_math = readmatrix("survey_response.csv");
% value: 1 [Agree a lot], 2 [Agree a little], 3 [Disagree a little], 4 [Disagree a lot]

mean_vec_2 = zeros(4, K2);
mean_vec_1 = zeros(4, K1);

for a = 1:4
    ind_a = find(like_math(:,c) == a);
    mean_vec_2(a,:) = mean(A2_est(ind_a, :));
    mean_vec_1(a,:) = mean(A1_est(ind_a, :));
end

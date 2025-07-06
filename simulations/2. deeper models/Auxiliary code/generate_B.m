function [B1] = generate_B(J, K1, max_val)
% Generate the strictly identifiable J x (K1+1) coefficient matrix B1
% @param J: number of rows in B1
% @param K1: number of columns in B1 minus one
% @param max_val: default nonzero values in B1

B1_sub = zeros(J, K1);
for k = 1:K1
    if k+floor(K1/2) <= K1
        B1_sub([k+2*K1], k) = max_val;
        B1_sub(k+2*K1+floor(K1/2), k) = -max_val/2;
        B1_sub(k+2*K1, k+floor(K1/2)) = -max_val/2;
        B1_sub([k+2*K1+floor(K1/2)], k+floor(K1/2)) = max_val;
    end
    B1_sub([k k+K1], k) = max_val;
end
B1 = [[-max_val/2*ones(K1,1); -max_val*ones(K1,1); -max_val/2*ones(K1,1)], B1_sub];
end
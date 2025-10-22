function [ SLW_LRRSTO_out ] = func_SLW_LRRSTO( hsi,A,beta,lambda)
% ¿ÕÆ×½áºÏ
% min |Z|_* + beta*||I*Z-ZV||_F^2 + lambda*|E|_2,1 s.t., X = AZ+E; I= [I1,I2...,I9]; ZV = [ZV1,...,ZV9];

[N,NN,M]=size(hsi);

[Z,E] = lrra_SS_one_SW_G(hsi,A,beta,lambda);     % % min |Z|_* + beta*||I*Z-ZV||_F^2 + lambda*|E|_2,1 s.t., X = AZ+E 1d'*Z = 1n, I= [I1,I2...,I9];  ZV = [ZV1,...,ZV9];

SLW_LRRSTO_out = zeros(1,N*NN);
for i = 1: N*NN
   SLW_LRRSTO_out(:,i) = norm(E(:,i));    
end
SLW_LRRSTO_out = reshape(SLW_LRRSTO_out',N,NN);

end



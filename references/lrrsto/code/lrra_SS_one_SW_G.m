function [Z,E] = lrra_SS_one_SW_G(X,A,beta,lambda)
% This routine solves the following nuclear-norm optimization problem,
% which is more general than "lrr.m"
% min |Z|_*+lambda*|E|_2,1 s.t., X = AZ+E 1d'*Z = 1n
% 添加空间信息：min |Z|_* + beta*||I*Z-ZV||_F^2 + lambda*|E|_2,1 s.t., X = AZ+E 1d'*Z = 1n
% min |J|_* + beta*||I*Z-JV||_F^2 + lambda*|E|_2,1 s.t., X = AZ+E; Z = J; 1d'*Z = 1n
% 9个滑动窗口，I= [I,...,I];  JV = [JV,...,JV];
% <Spectral–Spatial Sparse Subspace Clustering for Hyperspectral Remote Sensing Images>
% <A spectral-spatial based local summation anomaly detection method for hyperspectral images>
[N,NN,M] = size(X);
X = reshape(X,N*NN,M);
X = X';

if nargin<4
    lambda = 1;
end
tol = 1e-8;
maxIter = 1e6;
[d n] = size(X);
m = size(A,2);
n2 = min(m,n);  % Z范数求解
rho = 1.1;
normfX = norm(X,'fro');
tol2 = 1e-4;
opt.tol = tol2;%precision for computing the partial SVD
opt.p0 = ones(m,1);

% 修改mu,参考inexact_alm_rpca_mc（）
norm_two = lansvd(X, 1, 'L');
mu = 1.25 / norm_two; % this one can be tuned
max_mu = mu * 1e9;
sv = 5;
svp = sv;

%% Initializing optimization variables
% intialize
J = zeros(m,n);
Z = zeros(m,n);
Y2 = zeros(m,n);

E = sparse(d,n);
Y1 = zeros(d,n);

% [m n] = size(Z);
ed =ones(m,1); 
en = ones(n,1);
Y3 = zeros(n,1)';
%% Start main loop
iter = 0;
disp(['initial,rank=' num2str(rank(Z))]);
while iter<maxIter
    iter = iter + 1;
    Ek = E;
    Zk = Z;
    Jk = J;
    
    %%    update J
    temp = Z + Y2/mu;
    %     [U,sigma,V] = lansvd(temp,m,n,sv,'L',opt);
    [U,sigma,V] = lansvd(temp,sv,'L',opt);             %%%computes the K largest singular values.
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    
    if svp < sv
        sv = min(svp + 1, n2);
    else
        sv = min(svp + round(0.05*n2), n2);
    end
    
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
        
    %% update JV
    % 通用模式
    JV0 = reshape(J',N,NN,m);
    JV = zeros(N,NN,m);
    
    
    % N*N窗口平滑滤波 （若是2的话，就是3*3的右下角）
    N_ft = 3;
    filter_W = ones(N_ft)/(N_ft*N_ft-1);
    cen_ft = (N_ft+1)/2;
    filter_W(cen_ft,cen_ft)=0;
    for i = 1:m
%         JV(:,:,i) = filter2(fspecial('average',3),JV0(:,:,i));  3*3窗口所有像素求均值
        JV(:,:,i) = filter2(filter_W,JV0(:,:,i));   % 3*3窗口只有邻域所有像素求均值
    end
% 
% %     % 上下左右平滑约束（效果不太好，L2:0.98,rx:0.94）
% %     smooth_kernel = [0,1,0;1,0,1;0,1,0]/4;
% %     for i = 1:m
% %         JV(:,:,i) = filter2(smooth_kernel,JV0(:,:,i));
% %     end
%     
% %     % 上下左右中平滑约束（效果不太好，L2:0.98,rx:0.94）
% %     smooth_kernel = [0,1,0;1,1,1;0,1,0]/5;
% %     for i = 1:m
% %         JV(:,:,i) = filter2(smooth_kernel,JV0(:,:,i));
% %     end

    JV = reshape(JV,n,m);
    JV = JV';
    


    %%  udpate Z
    ata = (2*beta+mu)*eye(m) + mu*(A'*A)+mu*(ed*ed');
    inv_a = inv(ata);
    Z = inv_a*( 2*beta*JV + mu*A'*(X-E+Y1/mu) + mu*(J-Y2/mu) + mu*ed*(en'-Y3/mu) );    
    
%     Z = max(Z,0);  % 2017-2-15 非零约束
    
    %%  update E
    temp = X-A*Z+Y1/mu;
    E = solve_l1l2(temp,lambda/mu);
    
    leq1 = X-A*Z-E;
    leq2 = Z-J;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    
    relChgZ = norm(Z-Zk,'fro')/normfX;
    relChgE = norm(E-Ek,'fro')/normfX;
    relChgJ = norm(J-Jk,'fro')/normfX;
    
    relChgZJ = norm(Z-J,'fro')/normfX;
    relChg = max(relChgE,max(relChgZ,relChgJ));
    recErr = norm(leq1,'fro')/normfX;
    
%     disp(['iter ' num2str(iter) ',mu=' num2str(mu) ...
%             ',rank=' num2str(svp) ',stopALM=' num2str(stopC) ',relChg=' num2str(relChg) ',recErr=' num2str(recErr) ',relChgZJ=' num2str(relChgZJ)]);
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
%     if stopC<tol 
%         break;
    if recErr<1e-5  &&  relChg<1e-2
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end

function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end

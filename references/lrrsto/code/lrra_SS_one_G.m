function [Z,E] = lrra_SS_one_G(X,A,beta,lambda)
%
% This routine solves the following nuclear-norm optimization problem,
% which is more general than "lrr.m"
% min |Z|_*+lambda*|E|_2,1 s.t., X = AZ+E 1d'*Z = 1n
% ��ӿռ���Ϣ��min |Z|_* + beta*||I*Z-ZV||_F^2 + lambda*|E|_2,1 s.t., X = AZ+E 1d'*Z = 1n
% min |J|_* + beta*||I*Z-JV||_F^2 + lambda*|E|_2,1 s.t., X = AZ+E; Z = J; 1d'*Z = 1n
% 9���������ڣ�I= [I,...,I];  JV = [JV,...,JV];
% <Spectral�CSpatial Sparse Subspace Clustering for Hyperspectral Remote Sensing Images>
% <A spectral-spatial based local summation anomaly detection method for hyperspectral images>
%
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
n2 = min(m,n);  % Z�������
rho = 1.1;
normfX = norm(X,'fro');
tol2 = 1e-4;
opt.tol = tol2;%precision for computing the partial SVD
opt.p0 = ones(m,1);

% �޸�mu,�ο�inexact_alm_rpca_mc����
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
    % HYDICE 80*100*d
%     JV0 = reshape(J',80,100,m);
%     JV = zeros(80,100,m);
%     % AVIRIS 100*100*d
%     JV0 = reshape(J',100,100,m);
%     JV = zeros(100,100,m);
%     % HyMap 200*200*d
%     JV0 = reshape(J',200,200,m);
%     JV = zeros(200,200,m);

% ͨ��ģʽ
    JV0 = reshape(J',N,NN,m);
    JV = zeros(N,NN,m);
    
    
%     % N*N����ƽ���˲� ������2�Ļ�������3*3�����½ǣ�
%     for i = 1:m
%         JV(:,:,i) = filter2(fspecial('average',2),JV0(:,:,i));
%     end
% 
% %     % ��������ƽ��Լ����Ч����̫�ã�L2:0.98,rx:0.94��
% %     smooth_kernel = [0,1,0;1,0,1;0,1,0]/4;
% %     for i = 1:m
% %         JV(:,:,i) = filter2(smooth_kernel,JV0(:,:,i));
% %     end
%     
% %     % ����������ƽ��Լ����Ч����̫�ã�L2:0.98,rx:0.94��
% %     smooth_kernel = [0,1,0;1,1,1;0,1,0]/5;
% %     for i = 1:m
% %         JV(:,:,i) = filter2(smooth_kernel,JV0(:,:,i));
% %     end
% 
%     JV = reshape(JV,n,m);
%     JV = JV';
    

    % N*N����ƽ���˲� 
    Js = 3; % ����
    Os = (Js+1)/2; % ż��
    gridx = 1:1 : Js-Os+1;
    if gridx(end)+Os-1 < Js
        grid = Js-Os+1;
        gridx = [gridx grid];
    elseif gridx(end)+Os-1>Js
       error('segment error!'); 
    end
    
    gridy = 1:1 : Js-Os+1;
    if gridy(end)+Os-1 < Js
        grid = Js-Os+1;
        gridy = [gridy grid];
    elseif gridy(end)+Os-1>Js
       error('segment error!'); 
    end

    lx = length(gridx);
    ly = length(gridy);
    JV_Os = [];
    I_Os = [];
    I = eye( size(Z,1) );
    for k = 1:lx*ly
        cov_kel = zeros(Js,Js);
        ii = floor((k-1)/ly)+1;     %�к�
        jj = mod(k-1,ly)+1;    %�к�
        xx = gridx(ii);     % ��
        yy = gridy(jj);     % ��
        % cov_kel(xx:xx+Os-1, yy:yy+Os-1) = 1/(Os^2);
        cov_kel(xx:xx+Os-1, yy:yy+Os-1) = 1/(Os^2-1);
        cov_kel(Os,Os) = 0;
        for i = 1:m
            JV(:,:,i) = filter2(cov_kel,JV0(:,:,i));
        end
        JV_T = reshape(JV,n,m);
        JV_Os = [JV_Os;JV_T'];
        I_Os = [I_Os;I];
    end
    

    %%  udpate Z
    ata = 2*beta*(I_Os'*I_Os) + mu*eye(m) + mu*(A'*A)+mu*(ed*ed');
    inv_a = inv(ata);
    Z = inv_a*( 2*beta*I_Os'*JV_Os + mu*A'*(X-E+Y1/mu) + mu*(J-Y2/mu) + mu*ed*(en'-Y3/mu) );    
    
%     Z = max(Z,0);  % 2017-2-15 ����Լ��
    
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

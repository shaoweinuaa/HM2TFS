function [H,Lh] = cons_hypergraph(X,k)
m = size(X,1);
H = zeros(m,m);    %H是incidence matrix;
for j = 1:m              %利用KNN算法构建超图，与样本欧式距离最近的K个点构成一个超边
    z = knn(X,X(j,:),k);  %knn中的K变化时，De也要变化；
    for i = 1:k
         H(z(i),j) = 1;
    end
end
A = eye(m);    % A是hyperedge weights对角矩阵，每条超边的权重置为1；
Dv = zeros(m,m);   %Dv是表示各顶点的度的对角矩阵；
tmp1 = sum(H'*A,2);
for i = 1:m
    Dv(i,i) = tmp1(i);
end
De = k*eye(m);  %De是表示各超边的度的对角矩阵；
theta1 = (Dv^-0.5)*H*A*(De^-1)*H'*(Dv^-0.5);
I = eye(m);
Lh = I - theta1;    % LapLacian matrix;

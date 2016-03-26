%拟牛顿法中BFGS算法求解f = x1*x1+2*x2*x2-2*x1*x2-4*x1的最小值，起始点为x0=[1 1]  H0为二阶单位阵
%算法根据最优化方法（天津大学出版社）122页编写
%v1.0 author: liuxi BIT

%format long
syms  x1 x2 alpha;
f = x1*x1+2*x2*x2-2*x1*x2-4*x1;%要最小化的函数
df=jacobian(f,[x1 x2]);%函数f的偏导
epsilon=1e-6;%0.000001
k=1;
x0=[1 1];%起始点
xk=x0;
gk=subs(df,[x1 x2],xk);%起始点的梯度
%gk=double(gk);
H0=[1 0;0 1];%初始矩阵为二阶单位阵
while(norm(gk)>epsilon)%迭代终止条件||gk||<=epsilon
     if k==1
        pk=-H0*gk';%负梯度方向
        Hk0=H0;%HK0代表HK(k-1)
     else
        pk=-Hk*gk';
        Hk0=Hk;%HK0代表HK(k-1)
    end 
    f_alpha=subs(f,[x1 x2],xk+alpha*pk');%关于alpha的函数
    [left right] = jintuifa(f_alpha,alpha);%进退法求下单峰区间
    [best_alpha best_f_alpha]=golddiv(f_alpha,alpha,left,right);%黄金分割法求最优步长
    xk=xk+best_alpha*pk';
    gk0=gk;%gk0代表g(k-1)
    gk=subs(df,[x1 x2],xk);
    %gk=double(gk);
    yk=gk-gk0;
    sk=best_alpha*pk';%sk=x(k+1)-xk
    %====begin=============与DFP算法不同的地方==============
    wk=(yk*Hk*yk')^0.5*(sk'/(yk*sk')-Hk*yk'/(yk*Hk*yk'));
    Hk=Hk0-Hk0*yk'*yk*Hk0/(yk*Hk0*yk')+sk'*sk/(yk*sk')+wk*wk';%修正公式
    %====end===============与DFP算法不同的地方==============
    k=k+1;
end
best_x=xk%最优点
best_fx=subs(f,[x1 x2],best_x)%最优值
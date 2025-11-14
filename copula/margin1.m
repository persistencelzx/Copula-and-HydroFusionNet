
%% 寻找最优边缘分布
function [ycdf1,ycdf2,Pd1,Pd2]=margin1(x1,x2)
n=length(x1);
% 常见分布参数估计
pd1 = fitdist(x1,'normal');   %对数正态分布参数拟合
pd2 = fitdist(x1,'GeneralizedExtremeValue'); %gev分布参数拟合
cdf1=[x1,cdf(pd1,x1)];    %计算理论频率P; 
cdf2=[x1,cdf(pd2,x1)]; 
% 调用kstest函
[h1,p1,ksstat_pot1,cv1]=kstest(x1,cdf1);%ksstat1是ks值
[h2,p2,ksstat_pot2,cv2]=kstest(x1,cdf2);%ksstat1是ks值
ks_value=[ksstat_pot1;ksstat_pot2];
[mm,nn]=min(ks_value);
if nn==1
    ycdf1=cdf(pd1,x1);
    Pd1=pd1;
    disp('变量1最优边缘分布为正态分布');
elseif nn==2
    ycdf1=cdf(pd2,x1);
    Pd1=pd2;
    disp('变量1最优边缘分布为GEV分布');
end

% 常见分布参数估计
pd1 = fitdist(x2,'normal');   %对数正态分布参数拟合
pd2 = fitdist(x2,'GeneralizedExtremeValue'); %gev分布参数拟合
cdf1=[x2,cdf(pd1,x2)];    %计算理论频率P; 
cdf2=[x2,cdf(pd2,x2)]; 
% 调用kstest函
[h1,p1,ksstat_pot1,cv1]=kstest(x2,cdf1);%ksstat1是ks值
[h2,p2,ksstat_pot2,cv2]=kstest(x2,cdf2);%ksstat1是ks值
ks_value=[ksstat_pot1;ksstat_pot2];
[mm,nn]=min(ks_value);
if nn==1
    ycdf2=cdf(pd1,x2);
    Pd2=pd1;
    disp('变量2最优边缘分布为正态分布');
elseif nn==2
    ycdf2=cdf(pd2,x2);
    Pd2=pd2;
    disp('变量2最优边缘分布为GEV分布');
end
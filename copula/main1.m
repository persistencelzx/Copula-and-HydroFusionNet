clear;clc
%% 1 数据读取与整合处理
a=xlsread('data1.xlsx'); %读取所有数据
x1=a(:,2);%读取第二例数据
x2=a(:,23);%读取第三例数据
n=length(x1);
[ycdf1,ycdf2,Pd1,Pd2]=margin1(x1,x2);
U=ycdf1;
V=ycdf2;
% 调用copulafit函数估计二元Gaussian-Copula中的参数
rho_Gaussian = copulafit('Gaussian',[U(:), V(:)]);
% 调用copulafit函数估计二元t-Copula中的参数
[rho_t,nuhat] = copulafit('t',[U(:), V(:)]);
% 调用copulafit函数估计二元Frank-Copula中的参数
rho_Frank = copulafit('Frank',[U(:), V(:)]);
% 调用copulafit函数估计二元Gumbel-Copula中的参数
rho_Gumbel = copulafit('Gumbel',[U(:), V(:)]);
% 调用copulafit函数估计二元Clayton-Copula中的参数
rho_Clayton = copulafit('Clayton',[U(:), V(:)]);
%% 求5种copula的AIC和BIC值
% 计算Gaussian-Copula对应的AIC和BIC值
parameter_Gaussian=1;
PDF_Gaussian=copulapdf('Gaussian',[U(:), V(:)],rho_Gaussian);
CDF_Gaussian=copulacdf('Gaussian',[U(:), V(:)],rho_Gaussian);
% AIC_gaussian=-2*sum(log(CDF_Gaussian))+2*parameter_Gaussian;
BIC_gaussian=-2*sum(log(CDF_Gaussian))+log(n)*parameter_Gaussian;

% 计算t-Copula对应的AIC和BIC值
parameter_t=2; %还有个自由度参数，所以是2
PDF_t=copulapdf('t',[U(:), V(:)],rho_t,nuhat);
CDF_t=copulacdf('t',[U(:), V(:)],rho_t,nuhat);
BIC_t=-2*sum(log(CDF_t))+log(n)*parameter_t;
% AIC_t=-2*sum(log(CDF_t))+2*parameter_t;

% 计算Frank-Copula对应的AIC和BIC值
parameter_Frank=1;
PDF_Frank=copulapdf('Frank',[U(:), V(:)],rho_Frank);
CDF_Frank=copulacdf('Frank',[U(:), V(:)],rho_Frank);
% AIC_Frank=-2*sum(log(CDF_Frank))+2*parameter_Frank;
BIC_Frank=-2*sum(log(CDF_Frank))+log(n)*parameter_Frank;

% 计算Gumbel-Copula对应的AIC和BIC值
parameter_Gumbel=1;
PDF_Gumbel=copulapdf('Gumbel',[U(:), V(:)],rho_Gumbel);
CDF_Gumbel=copulacdf('Gumbel',[U(:), V(:)],rho_Gumbel);
% AIC_Gumbel=-2*sum(log(CDF_Gumbel))+2*parameter_Gumbel;
BIC_Gumbel=-2*sum(log(CDF_Gumbel))+log(n)*parameter_Gumbel;
 
% 计算Clayton-Copula对应的AIC和BIC值
parameter_Clayton=1;
PDF_Clayton=copulapdf('Clayton',[U(:), V(:)],rho_Clayton);
CDF_Clayton=copulacdf('Clayton',[U(:), V(:)],rho_Clayton);
% AIC_Clayton=-2*sum(log(CDF_Clayton))+2*parameter_Clayton;
BIC_Clayton=-2*sum(log(CDF_Clayton))+log(n)*parameter_Clayton;

% 调用ecdf函数求X和Y的经验分布函数
[fx, xsort] = ecdf(x1);
[fx1, x1sort] = ecdf(x2);
% 调用spline函数，利用样条插值法求原始样本点处的经验分布函数值
U1 = spline(xsort(2:end),fx(2:end),x1);
V1 = spline(x1sort(2:end),fx1(2:end),x2);
% 定义经验Copula函数C(u,v)
C = @(u,v)mean((U1 <= u).*(V1 <= v));

% 通过循环计算经验Copula函数在原始样本点处的函数值
CUV = zeros(size(U(:)));
for i=1:numel(U)
    CUV(i) = C(U1(i),V1(i));
end
% 计算二元Gaussian-Copula函数在原始样本点处的函数值
C_Gaussian = copulacdf('Gaussian',[U(:), V(:)],rho_Gaussian);
% 计算二元t-Copula函数在原始样本点处的函数值
C_t = copulacdf('t',[U(:), V(:)],rho_t,nuhat);
% 计算二元Frank-Copula函数在原始样本点处的函数值
C_Frank = copulacdf('Frank',[U(:), V(:)],rho_Frank);
% 计算二元Gumbel-Copula函数在原始样本点处的函数值
C_Gumbel = copulacdf('Gumbel',[U(:), V(:)],rho_Gumbel);
% 计算二元Clayton-Copula函数在原始样本点处的函数值
C_Clayton = copulacdf('Clayton',[U(:), V(:)],rho_Clayton);

% 计算5种copula的平方欧氏距离
d2_Gaussian = (CUV-C_Gaussian)'*(CUV-C_Gaussian);
d2_t = (CUV-C_t)'*(CUV-C_t);
d2_Frank = (CUV-C_Frank)'*(CUV-C_Frank);
d2_Gumbel = (CUV-C_Gumbel)'*(CUV-C_Gumbel);
d2_Clayton = (CUV-C_Clayton)'*(CUV-C_Clayton);
RMSE_1=sqrt(d2_Clayton/n);
RMSE_2=sqrt(d2_Frank/n);
RMSE_3=sqrt(d2_Gaussian/n);
RMSE_4=sqrt(d2_Gumbel/n);
RMSE_5=sqrt(d2_t/n);

%% 通过最小的BIC和最小平方欧式距离选择最优的Copula为Gaussian
%% 基于Copula的联合累积概率分布  Model1
x11 = linspace(min(x1)-0.5,max(x1)+0.5,100);
x22 = linspace(min(x2)-0.5,max(x2)+0.5,100);
ZZ = zeros(100,100);
for i = 1:100
    for j = 1:100
        u1=cdf(Pd1,x11(i));
        u2=cdf(Pd2,x22(j));
        ZZ(j,i) = copulacdf('Frank',[u1,u2],rho_Frank);%% 如果是Frank，则相应改成Frank
        PP(j,i)= copulapdf('Frank',[u1,u2],rho_Frank);%% 如果是Frank，则相应改成Frank
        RR(j,i)=1/(1-copulacdf('Frank',[u1,u2],rho_Frank));%% 如果是Frank，则相应改成Frank
        Tongxian(j,i)=(1-u1-u2+copulacdf('Frank',[u1,u2],rho_Frank));%% 如果是Frank，则相应改成Frank
        Tongxian_RP(j,i)=1/(1-u1-u2+copulacdf('Frank',[u1,u2],rho_Frank));%% 如果是Frank，则相应改成Frank
    end
end
[XX,YY] = meshgrid(x11,x22);
a1=[0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]; %等值线设置，可自定义
[C1, h1] = contour(YY, XX, RR, a1, 'ShowText', 'on');

title('概率CDF函数')
xlabel('干旱强度 Drought intensity');
ylabel('高温强度 heat intensity');
figure(2);
surf(XX,YY,ZZ);
s = surf(XX, YY, ZZ);
s.EdgeColor = 'none';   % 不显示边框线


title('联合概率CDF');
xlabel('干旱强度 Drought intensity');
ylabel('高温强度 heat intensity');
set(gca, 'LineWidth', 1.5);


figure(3) %%求概率密度函数图
surf(XX,YY,PP);
title('联合概率密度PDF');
xlabel('干旱强度 Drought intensity');
ylabel('高温强度 heat intensity');

figure(4);  % 创建图形窗口

% 生成网格数据
[XX, YY] = meshgrid(x11, x22);

% 设置等值线级别
a1 = [200 100 80 60 50 40 30 20 10 5];

% 绘制等高线图，并设置线条宽度为 2
[C1, h1] = contour(YY, XX, RR, a1, 'ShowText', 'on');
set(h1, 'LineWidth', 2);

% 设置标题和轴标签
title('联合重现期');
xlabel('干旱强度 Drought intensity');
ylabel('高温强度 Heat intensity');
set(gca, 'LineWidth', 1.5);

% 设置坐标轴范围从 (0,0) 开始
xlim([0, max(x11)]);
ylim([0, max(x22)]);


figure(5) %%求同现概率
[XX,YY] = meshgrid(x11,x22);
a1=[0.2 0.1 0.05 0.02 0.01]; %等值线设置，可自定义
[C1,h1]=contour(YY,XX,Tongxian,a1,'Showtext','on');
title('同现概率');
xlabel('干旱强度 Drought intensity');
ylabel('高温强度 heat intensity');

figure(6) %%求同现重现期
[XX,YY] = meshgrid(x11,x22);
a1=[200 100 50 20 10 ]; %等值线设置，可自定义
[C1,h1]=contour(YY,XX,Tongxian_RP,a1,'Showtext','on');
title('同现重现期');
xlabel('干旱强度 Drought intensity');
ylabel('高温强度 heat intensity');
clear; clc;

%% 1 数据读取
data = xlsread('data1.xlsx');
[nRows, nCols] = size(data);

% 每组变量对应：第2列配第23列，第3列配第24列，以此类推
numPairs = min(nCols - 22, 21);  % 最多配到第22列 & 第43列

% 结果表初始化
resultTable = table();

for i = 1:numPairs
    colX = i + 1;       % x 变量列（第2列开始）
    colY = i + 22;      % y 变量列（第23列开始）

    x1 = data(:, colX);
    x2 = data(:, colY);
    n = length(x1);

    % 经验分布边缘转换
    [ycdf1, ycdf2, ~, ~] = margin1(x1, x2);
    U = ycdf1;
    V = ycdf2;

    % Copula 拟合
    rho_Gaussian = copulafit('Gaussian', [U, V]);
    [rho_t, nuhat] = copulafit('t', [U, V]);
    rho_Frank = copulafit('Frank', [U, V]);
    rho_Gumbel = copulafit('Gumbel', [U, V]);
    rho_Clayton = copulafit('Clayton', [U, V]);

    % 经验分布函数估计
    [fx, xsort] = ecdf(x1);
    [fx1, x1sort] = ecdf(x2);
    U1 = spline(xsort(2:end), fx(2:end), x1);
    V1 = spline(x1sort(2:end), fx1(2:end), x2);

    % 经验 Copula 估计
    C = @(u,v) mean((U1 <= u) .* (V1 <= v));
    CUV = arrayfun(@(j) C(U1(j), V1(j)), 1:n)';

    % 拟合分布函数
    C_Gaussian = copulacdf('Gaussian', [U, V], rho_Gaussian);
    C_t = copulacdf('t', [U, V], rho_t, nuhat);
    C_Frank = copulacdf('Frank', [U, V], rho_Frank);
    C_Gumbel = copulacdf('Gumbel', [U, V], rho_Gumbel);
    C_Clayton = copulacdf('Clayton', [U, V], rho_Clayton);

    % BIC
    BIC_G = -2 * sum(log(C_Gaussian)) + log(n) * 1;
    BIC_t = -2 * sum(log(C_t)) + log(n) * 2;
    BIC_F = -2 * sum(log(C_Frank)) + log(n) * 1;
    BIC_Gu = -2 * sum(log(C_Gumbel)) + log(n) * 1;
    BIC_C = -2 * sum(log(C_Clayton)) + log(n) * 1;

    % RMSE
    RMSE_G = sqrt(mean((CUV - C_Gaussian).^2));
    RMSE_t = sqrt(mean((CUV - C_t).^2));
    RMSE_F = sqrt(mean((CUV - C_Frank).^2));
    RMSE_Gu = sqrt(mean((CUV - C_Gumbel).^2));
    RMSE_C = sqrt(mean((CUV - C_Clayton).^2));

    % Copula 参数整理
    rho_G = rho_Gaussian(1,2);
    rho_t_corr = rho_t(1,2);

    models = {'Gaussian', 't', 'Frank', 'Gumbel', 'Clayton'};
    rhos = [rho_G; rho_t_corr; rho_Frank; rho_Gumbel; rho_Clayton];
    dfs = [NaN; nuhat; NaN; NaN; NaN];
    BICs = [BIC_G; BIC_t; BIC_F; BIC_Gu; BIC_C];
    RMSEs = [RMSE_G; RMSE_t; RMSE_F; RMSE_Gu; RMSE_C];

    groupNames = repmat(string(['Pair_' num2str(colX) '_' num2str(colY)]), 5, 1);

    % 合并至总表格
    tempT = table(groupNames, models, rhos, dfs, BICs, RMSEs, ...
                  'VariableNames', {'Pair', 'Model', 'Rho', 'DegreeFreedom', 'BIC', 'RMSE'});
    resultTable = [resultTable; tempT];
end

% 保存结果
writetable(resultTable, 'Copula_Model_AllPairs.xlsx');
disp('所有配对的Copula模型结果已保存到 Copula_Model_AllPairs.xlsx');

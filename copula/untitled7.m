% 假设已有原始数据：
models = repmat({'Gaussian'; 't'; 'Frank'; 'Gumbel'; 'Clayton'}, 26, 1);  % 26组共130行
rhos = [...  % 按照你贴的顺序粘贴进去
    2.44E-01; 2.43E-01; 1.57E+00; 1.18E+00; 2.40E-01;
    3.27E-01; 3.27E-01; 2.30E+00; 1.29E+00; 2.51E-01;
    3.73E-01; 3.71E-01; 2.36E+00; 1.29E+00; 3.59E-01;
    4.77E-01; 4.74E-01; 3.31E+00; 1.45E+00; 5.22E-01;
    4.13E-01; 4.06E-01; 2.38E+00; 1.28E+00; 6.02E-01;
    4.53E-01; 4.47E-01; 2.62E+00; 1.36E+00; 6.64E-01;
    3.37E-01; 3.36E-01; 2.08E+00; 1.28E+00; 3.09E-01;
    3.59E-01; 3.56E-01; 2.36E+00; 1.20E+00; 4.54E-01;
    4.87E-01; 4.84E-01; 3.43E+00; 1.40E+00; 5.47E-01;
    5.39E-02; 5.39E-02; 4.40E-01; 1.05E+00; 1.45E-06;
    4.41E-01; 4.36E-01; 2.65E+00; 1.30E+00; 6.20E-01;
    4.60E-01; 4.66E-01; 3.29E+00; 1.42E+00; 6.45E-01;
    5.45E-01; 5.48E-01; 3.58E+00; 1.51E+00; 9.41E-01;
    4.77E-01; 4.76E-01; 3.70E+00; 1.38E+00; 6.36E-01;
    3.81E-01; 3.87E-01; 2.32E+00; 1.31E+00; 4.85E-01;
    3.51E-01; 3.50E-01; 2.19E+00; 1.30E+00; 3.15E-01;
    5.65E-01; 5.61E-01; 3.71E+00; 1.54E+00; 8.23E-01;
    4.33E-01; 4.35E-01; 2.73E+00; 1.29E+00; 6.94E-01;
    5.91E-01; 5.91E-01; 4.45E+00; 1.59E+00; 8.37E-01;
    4.52E-01; 4.58E-01; 2.90E+00; 1.42E+00; 5.75E-01;
    5.22E-01; 5.25E-01; 3.88E+00; 1.43E+00; 8.36E-01;
];

% 新顺序索引（Frank, Clayton, Gumbel, t, Gaussian）
newOrder = [3 5 4 2 1];

% 总组数
numGroups = length(rhos) / 5;

% 初始化新表格
newModels = cell(length(rhos), 1);
newRhos = zeros(length(rhos), 1);

for i = 1:numGroups
    baseIdx = (i-1)*5;
    groupModels = models(baseIdx+1 : baseIdx+5);
    groupRhos = rhos(baseIdx+1 : baseIdx+5);

    newGroupModels = groupModels(newOrder);
    newGroupRhos = groupRhos(newOrder);

    newModels(baseIdx+1 : baseIdx+5) = newGroupModels;
    newRhos(baseIdx+1 : baseIdx+5) = newGroupRhos;
end

% 构建新表格
T = table(newModels, newRhos, 'VariableNames', {'Model', 'Rho'});

% 可选：保存到文件
writetable(T, 'Reordered_Copulas.xlsx');

disp("Copula 顺序调整完成，并已保存为 Reordered_Copulas.xlsx");

function fit_distributions_loop()
    clear; clc;

    % 读取数据
    data = xlsread('data1.xlsx');
    num_cols = size(data, 2);
    
    % 初始化结果存储
    results = table();

    % 循环第 3 列到第 22 列（共 20 列）
    for col = 2:43
        x = data(:, col);
        x = x(~isnan(x)); % 去除 NaN

        % 如果数据量太少跳过
        if length(x) < 10
            warning("列 %d 数据太少，跳过。", col);
            continue;
        end

        % 拟合分布
        pd_norm = fitdist(x, 'normal');
        pd_gev = fitdist(x, 'GeneralizedExtremeValue');

        % KS 检验准备
        x_sorted = sort(x);
        cdf_norm = cdf(pd_norm, x_sorted);
        cdf_gev = cdf(pd_gev, x_sorted);

        % 构建 CDF 数据用于 kstest
        cdf_data_norm = [x_sorted, cdf_norm];
        cdf_data_gev = [x_sorted, cdf_gev];

        % KS 检验
        [~, p_norm, ks_norm] = kstest(x, cdf_data_norm);
        [~, p_gev, ks_gev] = kstest(x, cdf_data_gev);

        % RMSE 计算
        M = length(x_sorted);
        empirical = (1:M)' / (M + 1);
        rmse_norm = sqrt(mean((sort(cdf_norm) - empirical).^2));
        rmse_gev  = sqrt(mean((sort(cdf_gev) - empirical).^2));

        % 存储当前列结果
        temp = table(col, ...
            pd_norm.mu, pd_norm.sigma, ...
            pd_gev.k, pd_gev.sigma, pd_gev.mu, ...
            p_norm, ks_norm, rmse_norm, ...
            p_gev, ks_gev, rmse_gev, ...
            'VariableNames', {
                'ColumnIndex', ...
                'Norm_mu', 'Norm_sigma', ...
                'GEV_k', 'GEV_sigma', 'GEV_mu', ...
                'KS_p_Norm', 'KS_stat_Norm', 'RMSE_Norm', ...
                'KS_p_GEV', 'KS_stat_GEV', 'RMSE_GEV'});

        results = [results; temp];
    end

    % 写入 Excel
    writetable(results, 'Distribution_Fit_Results.xlsx');
    fprintf('所有分布拟合结果已保存为 Distribution_Fit_Results.xlsx\n');
end

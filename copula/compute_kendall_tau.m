function compute_kendall_tau() 
    % 读取 Excel 文件
    filename = 'data1.xlsx';
    data = xlsread(filename);
    
    % 获取数据的列数
    num_columns = size(data, 2);
    
    % 创建一个空的表格来存储结果
    results = table();
    
    % 循环提取指定列对进行 Kendall's tau 检验
    for i = 2:22  % i 从 2 开始，最多到第 22 列
        % 提取第 i 列和第 i+21 列的数据（比如：第 2 列和第 23 列）
        x = data(:, i);    % 当前列（如：第2列）
        y = data(:, i+21); % 对应的列（如：第23列）
        
        % 去除NaN值（如果存在缺失）
        valid_idx = ~isnan(x) & ~isnan(y);
        x = x(valid_idx);
        y = y(valid_idx);
        
        % 计算 Kendall's tau 相关性
        [tau, p] = corr(x, y, 'type', 'Kendall');
        
        % 判断显著性
        if p < 0.05
            significance = "Significant (p < 0.05)";
        else
            significance = "Not Significant (p >= 0.05)";
        end
        
        % 将当前列对的结果添加到表格中
        temp_table = table(i, i+21, tau, p, significance, ...
            'VariableNames', {'Column_i', 'Column_i+21', 'Kendall_tau', 'p_value', 'Significance'});
        
        % 拼接到结果表格中
        results = [results; temp_table];
    end
    
    % 保存结果到 Excel 文件
    output_filename = 'kendall_tau_results.xlsx';
    writetable(results, output_filename);
    
    % 显示完成信息
    fprintf('Results have been saved to %s\n', output_filename);
end

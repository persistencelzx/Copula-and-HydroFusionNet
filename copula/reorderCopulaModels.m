function reorderCopulaModels(filenameIn, filenameOut)
    % 从 Excel 文件读取数据
    data = readtable(filenameIn);

    % 每5行为一组
    n = height(data);
    newData = table();  % 初始化输出表格
    for i = 1:5:n
        group = data(i:i+4, :);

        % 目标顺序
        order = {'Frank', 'Clayton', 'Gumbel', 't', 'Gaussian'};
        [~, idx] = ismember(order, group.Model);

        % 判断是否所有模型都存在，避免错误
        if any(idx == 0)
            warning('模型名称缺失于第 %d~%d 行，跳过该组', i, i+4);
            continue;
        end

        % 重新排序并添加
        newData = [newData; group(idx, :)]; %#ok<AGROW>
    end

    % 将结果写入 TXT 文件（制表符分隔）
    writetable(newData, 'copula_results_reordered.xlsx');

end

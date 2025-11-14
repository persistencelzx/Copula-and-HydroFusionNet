function reformatColumnsByFiveRows(inputFile, outputFile)
    % 读取原始数据
    data = readmatrix(inputFile);  % 自动识别分隔符
    [nRows, nCols] = size(data);

    if mod(nRows, 5) ~= 0
        error("数据行数不是5的倍数，无法重组。");
    end

    % 计算5行一组共有多少组
    numGroups = nRows / 5;

    % 初始化输出数据
    reformattedData = [];

    for g = 1:numGroups
        rows = data((g-1)*5 + 1 : g*5, :);  % 当前的5行块

        % 每一列提取为一行（5个值），堆叠成3行
        newRows = rows';         % 转置为 3×5，每行是一列的数据
        reformattedData = [reformattedData; newRows];  % 添加到总输出中
    end

    % 写入输出文件
    writematrix(reformattedData, 'copula_results_final.xlsx');

end

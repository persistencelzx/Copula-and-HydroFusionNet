function formatOutput5PerRow(vec)
    % 将列向量 vec 按每5个一行格式输出
    vec = vec(:);  % 保证是列向量
    n = length(vec);
    for i = 1:5:n
        row = vec(i:min(i+4, n));
        fprintf('%.2E\t', row);
        fprintf('\n');
    end
end


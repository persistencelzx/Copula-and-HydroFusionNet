% 假设 DegreeFreedom 和 sita 是两个列向量，长度分别为 n
DegreeFreedom = [
    4.67E+06; 4.67E+06; 1.26E+07; 1.33E+07; 1.23E+07;
    4.64E+01; 1.14E+07; 1.31E+07; 1.17E+07; 1.29E+07; 
    1.15E+07;1.41E+06; 8.05E+00; 1.13E+07; 1.24E+07; 
    1.29E+07; 1.79E+01; 1.66E+07; 7.41E+01; 8.37E+00; 
    1.30E+07
];

sita = [
    2.43E-01; 3.27E-01; 3.71E-01; 4.74E-01; 4.06E-01;
    4.47E-01; 3.36E-01; 3.56E-01; 4.84E-01; 5.39E-02;
    4.36E-01; 4.66E-01; 5.48E-01; 4.76E-01; 3.87E-01;
    3.50E-01; 5.61E-01; 4.35E-01; 5.91E-01; 4.58E-01;
    5.25E-01
];

% 检查长度是否一致
n = min(length(DegreeFreedom), length(sita));

% 交错合并
combined = reshape([DegreeFreedom(1:n)'; sita(1:n)'], [], 1);

% 显示结果
disp(combined);

% 可选：写入文件
writematrix(combined, 'Combined_DF_Sita.txt');

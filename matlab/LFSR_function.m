function q=LFSR_function(seed,g,n,cycle)

% LFSR序列输出

% q：    LFSR依次输出所得矩阵

% seed： LFSR种子

% g：    反馈系数

% n：    序列长度

% cycle：循环次数，即时钟周期数

 

%分别获取种子和反馈系数矩阵

%低位在前

seed_matrix=rot90(bitget(seed,n:-1:1))';

g_matrix=rot90(bitget(g,n:-1:1))';

 

%输出矩阵第一行为seed

q(1,1:n)=seed_matrix;

 

%依次输出每一个触发器的值

%反馈系数决定是否异或

for i=1:cycle

    q(i+1,1)=q(i,n);

    for j=2:n

        if(g_matrix(j)==1)

            q(i+1,j)=xor(q(i,j-1),q(i,n));

        else

            q(i+1,j)=q(i,j-1);

        end

    end

end

%将高位在前转换成低位在前

q=rot90(q)';
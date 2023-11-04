% @File    :   halton.m
% @Time    :   2021/9/10
% @Author  :   Dongxu Lv 
% @Version :   0.1
% @Contact :   lvdongxu@sjtu.edu.cn
% @License :   (C)Copyright 2020-forever , SJTU-DMNE
% @Desc    :   Halton sequence generator

function seq = halton(x, len)
% Input:  
%        x: Fixed-point binary number
%        len: length
% Output: 
%        seq: Persudo random sequence

    halton_n = halton_num(len);
    seq      = halton_n > x;

end


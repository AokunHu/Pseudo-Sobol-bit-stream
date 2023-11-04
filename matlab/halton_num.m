% @File    :   halton_num.m
% @Time    :   2021/9/10
% @Author  :   Dongxu Lv 
% @Version :   0.1
% @Contact :   lvdongxu@sjtu.edu.cn
% @License :   (C)Copyright 2020-forever , SJTU-DMNE
% @Desc    :   Halton number generator

function halton_num = halton_num(L)
    % Input:  
    %        L      : halton sequence length
    % Output: 
    %        halton_num: halton sequence, 1xL
    
        halton_n     = haltonset();
        halton_num   = halton_n(1:L);
    
    end
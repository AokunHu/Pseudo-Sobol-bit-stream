% @File    :   determi_sequence.m
% @Time    :   2021/9/10
% @Author  :   Dongxu Lv 
% @Version :   0.1
% @Contact :   lvdongxu@sjtu.edu.cn
% @License :   (C)Copyright 2020-forever , SJTU-DMNE
% @Desc    :   Deterministic sequence generator(1111100000)

function seq = determi_sequence(x, len)
% Input:  
%        x: Fixed-point binary number
%        len: length
% Output: 
%        seq: results of maxpooling with a range of [0,1]

    num = x * len;
    seq = [ones(1,num), zeros(1,len-num)];

end
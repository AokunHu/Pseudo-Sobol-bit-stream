% @File    :   sobol_FSM.m
% @Time    :   2021/9/12
% @Author  :   Dongxu Lv 
% @Version :   0.1
% @Contact :   lvdongxu@sjtu.edu.cn
% @License :   (C)Copyright 2020-forever , SJTU-DMNE
% @Desc    :   Sobol sequence generator based on FSM by Aokun Hu

function seq = sobol_FSM(x, len, src_num)
% Input:  
%        x:       Fixed-point binary number
%        len:     length
%        src_num: Source number of sobol
% Output: 
%        seq: Persudo random sequence, 1 x len

    FSM      = zeros(src_num, len);
    sobol_n  = sobol_num(10, len);
    bitwidth = log2(len);
    seq      = zeros(1, len);

%%%%% Build Finite State Machine look-up table
    for src = 1 : src_num
        for i = 1 : len
            for b = 1 : bitwidth
                if (sobol_n(src,i) >= 1/2^b) && (sobol_n(src,i) < 1/2^(b-1))
                    FSM(src,i) = bitwidth - b;
                end
            end
            if sobol_n(src,i) < 1/2^bitwidth
                FSM(src,i) = bitwidth;
            end
        end
    end

%%%%% Use the FSM LUT to get the bit in x
    x_bin = x.bin;
    for i = 1 : len
        if FSM(src_num,i) == bitwidth
            seq(i) = 0;
        else
            seq(i) = str2num(x_bin(bitwidth - FSM(src_num,i)));
        end
    end

    
end

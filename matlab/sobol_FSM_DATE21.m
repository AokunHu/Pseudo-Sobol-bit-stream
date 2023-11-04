% @File    :   sobol_FSM_DATE21.m
% @Time    :   2021/9/10
% @Author  :   Dongxu Lv 
% @Version :   0.1
% @Contact :   lvdongxu@sjtu.edu.cn
% @License :   (C)Copyright 2020-forever , SJTU-DMNE
% @Desc    :   Sobol sequence generator based on FSM proposed in DATE2021 
% @Reference: Asadi, Sina & Najafi, M. Hassan & Imani, Mohsen. (2021). A Low-Cost FSM-based Bit-Stream Generator for Low-Discrepancy Stochastic Computing. 10.23919/DATE51398.2021.9474143. 

function seq = sobol_FSM_DATE21(x, len, src_num)
% Input:  
%        x:       Fixed-point binary number
%        len:     length
%        src_num: Source number of sobol
% Output: 
%        seq: Persudo random sequence, src_num x len

    FSM      = zeros(src_num, len);
    sobol_n  = sobol_num(src_num, len);
    bitwidth = log2(len);
    seq      = zeros(src_num, len);

%%%%% Build Finite State Machine look-up table
    for src = 1 : src_num
        for i = 1 : len
            for b = 1 : bitwidth
                if (sobol_n(src,i) >= (2^(b-1) - 1)/2^(b-1)) && (sobol_n(src,i) < (2^b - 1)/2^b)
                    FSM(src,i) = bitwidth - b;
                end
            end
            if sobol_n(src,i) >= (2^b - 1)/2^b
                FSM(src,i) = bitwidth;
            end
        end
    end

%%%%% Use the FSM LUT to get the bit in x
    x_bin = x.bin;
    for src = 1 : src_num
        for i = 1 : len
            if FSM(src,i) == bitwidth
                seq(src,i) = 0;
            else
                seq(src,i) = str2num(x_bin(bitwidth - FSM(src,i)));
            end
        end
    end

end



function seq = sobol_Sedosobol(x, len, src_num)
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


%%%%% Rebuild sedosobol

    % for i = 1 : len
    %     k = log2(i) ;
    %     if k > 5 
    %         if k == floor(k)
    %             k = floor(k)-1;
    %         else
    %             k = floor (k);
    %         end
    %         tempbin = dectobin(sobol_n(src_num,i-2^k),bitwidth);
    %         if(tempbin(k+1) == 1)
    %             tempbin(k+1) = 0;
    %         else
    %             tempbin(k+1) = 1;
    %         end
    %         sobol_n(src_num,i) = bintodec(tempbin,bitwidth);
      
    %     end
    % end



    for i = 1 : len
        k = log2(i) ;
        if k > 4 
    
            if k == floor(k)
                m = floor(k)-1;
            else
                m = floor (k);
            end
    
            if((i-2^m)<=2^(m-1))
                tempbin = dectobin(sobol_n(src_num,i-2^m+2^(m-1)),bitwidth);
            else
                tempbin = dectobin(sobol_n(src_num,i-2^m-2^(m-1)),bitwidth);
            end
    
            if(tempbin(m+1) == 1)
                tempbin(m+1) = 0;
            else
                tempbin(m+1) = 1;
            end
            sobol_n(src_num,i) = bintodec(tempbin,bitwidth);
      
        end
    end

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

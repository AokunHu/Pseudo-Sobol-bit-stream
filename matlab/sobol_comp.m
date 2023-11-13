
function seq = sobol_comp(x, len, src_num)
% Input:  
%        x:       Fixed-point binary number
%        len:     length
%        src_num: sobol source number
% Output: 
%        seq: sobol sequence, src_num x len

    sobol_n = sobol_num(10, len);
    seq     = sobol_n > x;
    seq     = seq(src_num, :);


end

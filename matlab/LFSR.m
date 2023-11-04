function seq = LFSR(x, len, seed, g, n)
% Input:  
%        x: Fixed-point binary number
%        len: length
% Output: 
%        seq: Persudo random sequence
    cmp = rand(1,len);
    %LFSR begin%
    Q=LFSR_function(seed,g,n,len-1);
    cmp = zeros(1,len);
    for i=1:len
        for j=1:n
            cmp(i) = cmp(i) + Q(i,j) * (0.5^j);
        end
    end
    %LFSR end%
    seq = cmp < x;

end
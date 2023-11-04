

function resBin = sobolMACv3(x_seq_1, x_seq_2)
% Input:  
%        x_seq_1: contains n sequences with L length, nxL
%        x_seq_2: contains n sequences with L length, nxL
% Output: 
%        resBin: fixed-point results, nx1

%%%%% SC parameters
    seqLength = size(x_seq_1, 2);
    seqNum    = size(x_seq_1, 1);
    binLength = log2(seqLength) + 1;
    allLength = seqNum - 1 + binLength;

%%%%% MAC for each sequence
    res = zeros(seqNum, 1);
    for i = 1 : seqNum
        for j = 1 : seqLength
            res(i,1) = res(i,1) + (x_seq_1(i,j) & x_seq_2(i,j));
        end
    end
%     resBin = fi(sum(res) / seqLength, 1, allLength+1, log2(seqLength)) ;
      resBin = sum(res) / seqLength ;
end

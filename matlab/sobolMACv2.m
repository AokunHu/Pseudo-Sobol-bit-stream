
function resBin = sobolMACv2(x_seq_1, x_seq_2)
% Input:  
%        x_seq_1: contains n sequences with L length, nxL
%        x_seq_2: contains n sequences with L length, nxL
% Output: 
%        resBin: fixed-point results, nx1

%%%%% SC parameters
    seqLength  = size(x_seq_1, 2);
    seqNum     = size(x_seq_1, 1);
    seq16times = seqLength / 16;

%%%%% MAC for each sequence
    pSum = zeros(seqNum, seq16times);
    res = zeros(seqNum, 1);
    for i = 1 : seqNum
        for j = 1 : seq16times

            seq1Slice = x_seq_1(i, 1+(j-1)*16 : j*16); % Slice 16bits
            seq2Slice = x_seq_2(i, 1+(j-1)*16 : j*16); % Slice 16bits
            %%%%%%%%%% 1-16 bit %%%%%%%%%%
            %%if mod(j,4) == 1 || mod(j,4) == 0
            if mod(j,2) == 1
                pSum(i,j) = sobol12MUL1(seq1Slice, seq2Slice);
            else
                pSum(i,j) = sobol12MUL2(seq1Slice, seq2Slice);
            end
        end
        res(i,1) = fi(sum(pSum(i,:)), 1, 7+seq16times, 0);
    end
    resBin = fi(sum(res) / seqLength, 1, 7+seq16times+seqNum, log2(seqLength)) ;
end

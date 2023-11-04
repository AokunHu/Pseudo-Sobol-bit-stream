
function resBin = sobolMACv1(x_seq_1, x_seq_2)
% Input:  
%        x_seq_1: contains n sequences with L length, nxL
%        x_seq_2: contains n sequences with L length, nxL
% Output: 
%        resBin: fixed-point results, nx1

%%%%% SC parameters
    seqLength = size(x_seq_1, 2);
    seqNum    = size(x_seq_1, 1);
    seq8times = seqLength / 8;
    binLength = log2(seqLength) + 1;
    allLength = seqNum - 1 + binLength;

%%%%% MAC for each sequence
    pSum = zeros(seqNum, seq8times);
    res = zeros(seqNum, 1);
    for i = 1 : seqNum
        for j = 1 : seq8times
            seq1Slice = x_seq_1(i, 1+(j-1)*8 : j*8); % Slice 8bits
            seq2Slice = x_seq_2(i, 1+(j-1)*8 : j*8); % Slice 8bits
            adder1    = fi(seq2Slice(2) + seq2Slice(4) + seq2Slice(6) + seq2Slice(8), 0, 3, 0);
            if seq1Slice(2) == 1
                andAdder1 = adder1;
            else
                andAdder1 = 0;
            end
            and1 = seq1Slice(1) & seq2Slice(1);
            and3 = seq1Slice(3) & seq2Slice(3);
            and5 = seq1Slice(5) & seq2Slice(5);
            and7 = seq1Slice(7) & seq2Slice(7);
            adder2 = fi(and1 + and3 + and5 + and7, 0, 4, 0);
            pSum(i, j) = fi(adder2 + andAdder1, 0, binLength, 0);
        end
        res(i,1) = fi(sum(pSum(i, :)), 0, binLength, 0);
    end
    resBin = fi(sum(res) / seqLength, 1, allLength+1, log2(seqLength)) ;
end

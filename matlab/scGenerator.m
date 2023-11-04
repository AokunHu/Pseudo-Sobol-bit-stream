

function seq = scGenerator(x, seqLength, seqType, lfsr_param)
% Input:  
%        x:         Fixed-point binary number
%        SeqLength: Sequence length
%        seqType:   Sequence Type, 1 x typeNum
% Output: 
%        seq: Random sequence, src_num x len
%        lfsr_param: lfsr_param[1]:seed  lfsr_param[2]:g lfsr_param[3]:n

%%%%% Set fixed-point type
    if(~exist('lfsr_param','var'))
        lfsr_param = [5,3,5];  % 如果未出现该变量，则对其进行赋值
    end
    
    x_fixed = x;
    typeNum = size(seqType, 2);
    seq     = zeros(typeNum, seqLength);

%%%%% Sequence Generator
    for i = 1 : typeNum
        str = seqType(i);
        S   = regexp(str, '-', 'split');
        if S(1) == "Sobol"
            switch S(2)
                case "Comparator" % Sobol generator based on comparator
                    seq(i,:) = sobol_comp(x_fixed, seqLength, str2num(S(3))); 
                case "DATE2021" % Sobol generator based on FSM (DATE2021)
                    seq(i,:) = sobol_FSM_DATE21(x_fixed, seqLength, str2num(S(3))); 
                case "FSM" % Sobol generator based on FSM (Aokun Hu   bachelor dissertation)
                    seq(i,:) = sobol_FSM(x_fixed, seqLength, str2num(S(3))); 
                case "Sedosobol"
                    seq(i,:) = sobol_Sedosobol(x_fixed, seqLength, str2num(S(3))); 
            end
        else
            switch str
                case 'Uniform' % Uniform pseudo random sequence 
                    seq(i,:) = persudo_random(x_fixed, seqLength);
                case 'Deterministic' % Deterministic sequence, 1111100000
                    seq(i,:) = determi_sequence(x_fixed, seqLength);
                case 'Halton' % Halton sequence
                    seq(i,:) = halton(x_fixed, seqLength);
                case 'LFSR'
                    seq(i,:) = LFSR(x_fixed, seqLength, lfsr_param(1), lfsr_param(2),lfsr_param(3));
            end
        end
    end
end
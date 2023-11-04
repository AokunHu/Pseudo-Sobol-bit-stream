
clc
clear;

%% Experiment params
iteration     = 3000; % Trial number
samples       = 1;   % Sample number for each trial
seqType       = ["LFSR", "Sobol-FSM-1","Sobol-FSM-2","LFSR","LFSR"]; % Sequence type
%%seqType       = ["LFSR", "LFSR","Sobol-FSM-2"];
%%seqType       = ["Sobol-Sedosobol", "Sobol-DATE2021-1","LFSR"]; % Sequence type

MAE_fix   = zeros(1, 8); % fixed-point relative to float-point
MAE_sc1   = zeros(1, 8); % sc relative to float-point 
MAE_sc2   = zeros(1, 8); % sc relative to float-point
MAE_sc3   = zeros(1, 8); % sc relative to float-point
MAE_sc4   = zeros(1, 8); % sc relative to float-point
%%set lfsr parameter
 

for fixLength = 3 : 7
%     lfsr_param_1 = [5,3,8];
    lfsr_param_1 = [3,3,8];
    lfsr_param_2 = [7,4,8];
    lfsr_param_3 = [3,1,8];
    lfsr_param_4 = [4,8,8];
    lfsr_param_5 = [6,4,8];

    fixFracLength = fixLength;    % Fixed-point number fraction length
    seqLength     = 2^fixLength; % Sequence length, the default is 2^fixLength

    %% Trial loop
    AE_fix   = zeros(1, iteration); % fixed-point relative to float-point
    AE_sc1   = zeros(1, iteration); % sc relative to float-point
    AE_sc2   = zeros(1, iteration); % sc relative to float-point
    AE_sc3   = zeros(1, iteration); % sc relative to float-point
    AE_sc4   = zeros(1, iteration); % sc relative to float-point
    for i = 1 : iteration
        %%%%%%%%%% Test number generating %%%%%%%%%%
        float_1 = rand(1,samples);
        float_2 = rand(1,samples);
%         float_3 = zeros(1,samples)+0.5;

        % Fixed point
        fm = fimath('RoundingMethod', 'Round',...
                    'OverflowAction', 'Saturate',...
                    'ProductMode', 'FullPrecision',...
                    'SumMode', 'FullPrecision');
        fix_1 = fi(float_1, 0, fixLength, fixFracLength, fm);
        fix_2 = fi(float_2, 0, fixLength, fixFracLength, fm);
%         fix_3 = fi(float_3, 0, fixLength, fixFracLength, fm);

        % SC sequences
        sc_1  = zeros(samples, seqLength);
        sc_2  = zeros(samples, seqLength);
        sc_3  = zeros(samples, seqLength);
        sc_4  = zeros(samples, seqLength);
        sc_5  = zeros(samples, seqLength);
        for j = 1 : samples
            sc_1(j,:) = scGenerator(fix_1(j), seqLength, seqType(1), lfsr_param_1);
            sc_2(j,:) = scGenerator(fix_2(j), seqLength, seqType(2), lfsr_param_2);
            sc_3(j,:) = scGenerator(fix_2(j), seqLength, seqType(3), lfsr_param_3);
            sc_4(j,:) = scGenerator(fix_2(j), seqLength, seqType(4), lfsr_param_4);
            sc_5(j,:) = scGenerator(fix_2(j), seqLength, seqType(5), lfsr_param_5);
        end
%         sc_3(1,:) = scGenerator(fix_3(j), seqLength, seqType(3), lfsr_param_3);
%         sc_3(2,:) = scGenerator(fix_3(j), seqLength, seqType(3), lfsr_param_4);
%         sc_3(3,:) = scGenerator(fix_3(j), seqLength, seqType(3), lfsr_param_5);
        %%%%%%%%%% Float computing %%%%%%%%%%
        floatMulRes = sum(float_1 .* float_2);

        %%%%%%%%%% Fix computing %%%%%%%%%%
        fixMulRes = fi(sum(fix_1 .* fix_2), 1, fixLength*2+samples+1, fixFracLength*2, fm);
                                  
        %%%%%%%%%% SC sobolMACv1 computing %%%%%%%%%%
        %%scMulRes1 = fi(sobolMACv1(sc_1, sc_2), 1, fixLength*2+samples+1, fixFracLength*2, fm);
        scMulRes1 = fi(sobolMACv3(sc_1, sc_2), 1, fixLength*2+samples+1, fixFracLength*2, fm);
        %%%%%%%%%% SC sobolMACv2 computing %%%%%%%%%%
        %%scMulRes2 = fi(sobolMACv2(sc_1, sc_2), 1, 8+seqLength/16+samples, log2(seqLength), fm);
        scMulRes2 = fi(sobolMACv3(sc_1, sc_3), 1, fixLength*2+samples+1, fixFracLength*2, fm);
        scMulRes3 = fi(sobolMACv3(sc_1, sc_4), 1, fixLength*2+samples+1, fixFracLength*2, fm);
        scMulRes4 = fi(sobolMACv3(sc_1, sc_5), 1, fixLength*2+samples+1, fixFracLength*2, fm);
        %%%%%%%%%% SC MAC8in computing %%%%%%%%%%
        %%scMulRes3 = fi(MAC8in(sc_1, sc_3), 1, fixLength*2+samples+1, fixFracLength*2, fm);
        
        %%%%%%%%%% SC MAC5in computing %%%%%%%%%%
        %scMulRes4 = fi(MAC16in(sc_1, sc_2, sc_3), 0, fixLength, fixFracLength, fm);
        %scMulRes4 = MAC16in(sc_1, sc_2, sc_3);
%         scMulRes4 = MAC4in(sc_1, sc_2, sc_3);


        AE_fix(i)   = abs(floatMulRes - fixMulRes);
        AE_sc1(i)   = abs(floatMulRes - scMulRes1);
        AE_sc2(i)   = abs(floatMulRes - scMulRes2);
        AE_sc3(i)   = abs(floatMulRes - scMulRes3);
        AE_sc4(i)   = abs(floatMulRes - scMulRes4);
        %%AE_sc4(i)   = abs(floatMulRes/4 - scMulRes4);

    end

    MAE_fix(fixLength)   = sum(AE_fix) ./ iteration;
    MAE_sc1(fixLength)   = sum(AE_sc1) ./ iteration;
    MAE_sc2(fixLength)   = sum(AE_sc2) ./ iteration;
    MAE_sc3(fixLength)   = sum(AE_sc3) ./ iteration;
    MAE_sc4(fixLength)   = sum(AE_sc4) ./ iteration;
    %%MAE_sc4(fixLength-2)   = sum(AE_sc4) ./ iteration;
   
end

    MAE = [MAE_fix;MAE_sc1;MAE_sc2;MAE_sc3];
    len=[3,4,5,6,7];
    colormap Lines;
    Color = {[0 0 0],[0 0 1],[0 1 0],[1 0 0],[1 1 0],[1 0 1],[0 1 1],[1 0.2 0.2],[0.2 1 0.2],[0.2 0.2 1]};                          
    Marker = {'o','*','s','d','^','v','>','<','p','h'};
    figure(1);
    for j = 1 : 4
	    semilogy(len,MAE(j,3:7),'Color',Color{j},'LineStyle','-','Marker',Marker{j},'linewidth',1.5);
	    hold on;
    end
    grid on;
    xlabel('Resolution (2^N-bit length or N-bit width)');
    ylabel('MAE');
    legend('Binary','Psedu-Sobol','Sobol','LFSR');
%     title('level1');


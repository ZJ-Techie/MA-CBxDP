clc
clear all
% Data from the ADNI (adni.loni.usc.edu).
Y{1} = plasma_adj;
Y{2} = csf_adj;
Y{3} = img_vbm_adj;
Y{4} = FSresult;
z = BL_DX;
X = getNormalization(X,'normalize');
E = getNormalization(E,'normalize');
Y{1} = getNormalization(Y{1},'normalize');
Y{2} = getNormalization(Y{2},'normalize');
Y{3} = getNormalization(Y{3},'normalize');
Y{4} = getNormalization(Y{4},'normalize');
z = getNormalization(z,'normalize');

paras.MA_CBxDP.lambda.u1 = 1;
paras.MA_CBxDP.lambda.u2 = 0.1;  
paras.MA_CBxDP.lambda.u3 = 0.1;  
paras.MA_CBxDP.lambda.v1 = 0.1; 
paras.MA_CBxDP.lambda.v2 = 1; 
paras.MA_CBxDP.lambda.v3 = 1; 
paras.MA_CBxDP.lambda.v4 = 1; 
paras.MA_CBxDP.lambda.v5 = 1; 

Kfold = 5;
[n, ~] = size(X);
indices = crossvalind('Kfold', n, Kfold);
disp('Begin cross validition ...');
disp('==============================');
for k = 1 : Kfold
    fprintf('current fold: %d\n', k);
    test = (indices == k); 
    train = ~test;

    trainData.X = getNormalization(X(train, :),'normalize');
    trainData.E = getNormalization(E(train, :),'normalize');
    trainData.Y{1} = getNormalization(Y{1}(train, :),'normalize');
    trainData.Y{2} = getNormalization(Y{2}(train, :),'normalize');
    trainData.Y{3} = getNormalization(Y{3}(train, :),'normalize');
    trainData.Y{4} = getNormalization(Y{4}(train, :),'normalize');
    trainData.z = getNormalization(z(train, :),'normalize');

    testData.X = getNormalization(X(test, :),'normalize');
    testData.E = getNormalization(E(test, :),'normalize');
    testData.Y{1} = getNormalization(Y{1}(test, :),'normalize');
    testData.Y{2} = getNormalization(Y{2}(test, :),'normalize');
    testData.Y{3} = getNormalization(Y{3}(test, :),'normalize');
    testData.Y{4} = getNormalization(Y{4}(test, :),'normalize');
    testData.z = getNormalization(z(test, :),'normalize');

    tic

    [U_MA_CBxDP{k}, V_MA_CBxDP{k}, interceptQ{k}] = MA_CBxDP(trainData, paras.MA_CBxDP);
    
    toc

end
disp('==============================');


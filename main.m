% Burak Tokak
% https://github.com/btk/eigenfaces

% IMPORTANT: Please wait for the calculation to end, it might take about a
% munite depending on computer processing power.

% Make sure script is working in the same folder with datasets taken from
% http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html

clear all;
clc;
yale = load('Yale_32x32.mat')

% manipulate dataset for easier calculations
yale.fea = yale.fea'; % A'
yale.fea = yale.fea/255;
yale.gnd = yale.gnd'; % A'

% p=(2,3,4,5,6,7,8)
% 27 for p=2;
% 43 for p=3;
% 59 for p=4;
% 70 for p=5;
% 80 for p=6;
% 90 for p=7;
% 95 for p=8;

numOfBasisVecPerSplit = [
    [2, 27],
    [3, 43],
    [4, 59],
    [5, 70],
    [6, 80],
    [7, 90],
    [8, 95]
]

errRateTableNC = [];

for testSet = numOfBasisVecPerSplit'
    errRateTableNC(testSet(1)) = classifyErrorRate(yale, testSet(1), testSet(2), 'NC');
end

errRateTableNN = [];

for testSet = numOfBasisVecPerSplit'
    errRateTableNN(testSet(1)) = classifyErrorRate(yale, testSet(1), testSet(2), 'NN');
end


figure(1);
cdata = [errRateTableNC(2:end); errRateTableNN(2:end)];
xvalues = {'2','3','4','5','6','7','8'};
yvalues = {'NC','NN'};
h = heatmap(xvalues,yvalues,cdata);

h.Title = 'Error Rate %';
h.XLabel = 'Training Sets';
h.YLabel = 'Classification Algorithm';


function err_rate = classifyErrorRate(yale, p, numOfBasisVec, classifier)
    num_of_splits = 50;
    error_each_split = zeros(num_of_splits,1);

    % Load subset with the given id
    subsetId = p;
    subset = loadTrainData(subsetId);

    for n=1:num_of_splits

        fea_Train = yale.fea(:, subset(n).trainIdx);
        gnd_Train = yale.gnd(subset(n).trainIdx);
        fea_Test = yale.fea(:, subset(n).testIdx);
        gnd_Test = yale.gnd(subset(n).testIdx);

        % get mean of training data
        mean_face = mean(fea_Train,2);

        % remove mean
        meanRemoved_Train = fea_Train - repmat(mean_face,[1,size(fea_Train,2)]);

        % covariance matrix C=A'A
        C = meanRemoved_Train' * meanRemoved_Train;

        % eigenvectors and eigenvalues of C
        [eigenVector, eigenValue] = eig(C);

        % L=AA'
        eigenVectorL = meanRemoved_Train * eigenVector;

        all_eigenValue = diag(eigenValue);
        [val,idx_descendingOrder] = sort(all_eigenValue,'descend');
        eigenVectorL = eigenVectorL(:,idx_descendingOrder);
        eigenVectorL_K = eigenVectorL(:,1:numOfBasisVec);

        % normalize eigenfaces
        for j=1:numOfBasisVec
            eigenVectorL_K(:,j) = eigenVectorL_K(:,j)/norm(eigenVectorL_K(:,j));
        end

        % largest in decending order
        manip_train = transpose(eigenVectorL_K) * meanRemoved_Train;

        % Testing
        meanRemoved_Test = fea_Test - repmat(mean_face,[1,size(fea_Test,2)]);
        manip_test = transpose(eigenVectorL_K) * meanRemoved_Test;

        % classify faces relative to given classifier
        if(classifier == 'NC')
            retClasses = classifyNC(manip_test, manip_train, gnd_Train);
        else % NN
            retClasses = classifyNN(manip_test, manip_train, gnd_Train);
        end

        error_classify = sum(retClasses ~= gnd_Test);
        error_rate = (error_classify / size(fea_Test, 2))*100; % percentage
        error_each_split(n) = error_rate;
    end

    % avg
    err = sum(error_each_split)/num_of_splits;
    varr = sum((error_each_split-err).^2)/num_of_splits;
    err_rate = err
end


% Loads the train/test subset on a struct from folder with given p value
function t = loadTrainData(p)
    a = struct('trainIdx',{},'testIdx',{})
    for file = dir(sprintf('%dTrain/*.mat', p))'
        a(end+1) = load(sprintf('%dTrain/%s', p, file.name))
    end
    t = a
end

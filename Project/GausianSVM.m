clear all; close all; clc; 
glassesData = readtable('glasses.csv');
report = readtable('report.csv');

div = 100;

reduction = ceil(height(glassesData(:, 1))/div); % Divide data into groups of 100
arr = zeros(reduction, 9);

for i = 1:div:height(glassesData(:, 1)) % create cell for parsing each data entry
    formatIn = 'yyyy/mm/dd HH:MM:SS.FFF';
    arr(ceil(i/div), 1) = datenum(glassesData{i, 2},formatIn); % Date
    arr(ceil(i/div), 2) = glassesData{i, 3}; % Acc x
    arr(ceil(i/div), 3) = glassesData{i, 4}; % Acc y 
    arr(ceil(i/div), 4) = glassesData{i, 5}; % Acc z
    arr(ceil(i/div), 5) = glassesData{i, 6}; % Gyro x
    arr(ceil(i/div), 6) = glassesData{i, 7}; % Gyro y
    arr(ceil(i/div), 7) = glassesData{i, 8}; % Gyro z
    
    arr(ceil(i/div), 9) = sqrt(glassesData{i, 3}^2 + glassesData{i, 4}^2 + glassesData{i, 5}^2); % acc
    arr(ceil(i/div), 10) = sqrt(glassesData{i, 6}^2 + glassesData{i, 7}^2 + glassesData{i, 8}^2); % gyro
end



arr(:, 8) = 0; % Here we set the label for each class

C = categorical(report{:,2}); % Read all entries from column categorical in excel
types = categories(C); % Find the name of each category

for ti = 1:length(types) 
    t = types{ti};
    for i = 1:height(report)
        if strcmp(report{i,2}, t)
            from = datenum(report{i,4});
            to = datenum(report{i,5});
            filter = (arr(:, 1) <= to & arr(:, 1) >= from);
            arr(filter, 8) = 4;  % 4 means there is t
        end
    end
end

for i = 1:height(report)
    
    if strcmp(report{i,2}, 'Walk')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 2;
    end
    
    if strcmp(report{i,2}, 'Eat')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 1;
    end
    
    if strcmp(report{i,2}, 'Meeting')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 3;
    end
    
    if strcmp(report{i,2}, 'In computer')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 3;
    end
    
    if strcmp(report{i,2}, 'In vehicle')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 5;
    end
    
    if strcmp(report{i,2}, 'In bus')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 5;
    end
    
    if strcmp(report{i,2}, 'Train')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 5;
    end
    
end

figure(1)
x = 1:length(arr);
hold on
scatter(x(arr(:, 8) == 4), arr(arr(:, 8) == 4, 9), 'black', '.')
scatter(x(arr(:, 8) == 5), arr(arr(:, 8) == 5, 9), 'yellow', '.')
scatter(x(arr(:, 8) == 3), arr(arr(:, 8) == 3, 9), 'green', '.')
scatter(x(arr(:, 8) == 2), arr(arr(:, 8) == 2, 9), 'red', '.')
scatter(x(arr(:, 8) == 1), arr(arr(:, 8) == 1, 9), 'blue', '.')
legend('Idk','Vehicle/Bus/Train','Meeting/Computer','Walk','Eat');

figure(2)
x = 1:length(arr);
hold on
%scatter(arr(arr(:, 8) == 4, 10), arr(arr(:, 8) == 4, 9), 'black', '.')
scatter(arr(arr(:, 8) == 5, 10), arr(arr(:, 8) == 5, 9), 'yellow', '.')
scatter(arr(arr(:, 8) == 3, 10), arr(arr(:, 8) == 3, 9), 'green', '.')
scatter(arr(arr(:, 8) == 2, 10), arr(arr(:, 8) == 2, 9), 'red', '.')
scatter(arr(arr(:, 8) == 1, 10), arr(arr(:, 8) == 1, 9), 'blue', '.')
legend('Vehicle/Bus/Train','Meeting/Computer','Walk','Eat');

% Here to decide what data to take

figure(1) % Acc in z
x = 1:length(arr);
hold on
%scatter(arr(arr(:, 8) == 4, 10), arr(arr(:, 8) == 4, 9), 'black', '.')
scatter(x(arr(:, 8) == 5), arr(arr(:, 8) == 5, 4), 'yellow', '.');
scatter(x(arr(:, 8) == 3), arr(arr(:, 8) == 3, 4), 'green', '.');
scatter(x(arr(:, 8) == 2), arr(arr(:, 8) == 2, 4), 'red', '.');
scatter(x(arr(:, 8) == 1), arr(arr(:, 8) == 1, 4), 'blue', '.');
legend('Vehicle/Bus/Train','Meeting/Computer','Walk','Eat');
hold off

figure(2) % Gyro in x
x = 1:length(arr);
hold on
%scatter(arr(arr(:, 8) == 4, 10), arr(arr(:, 8) == 4, 9), 'black', '.')
scatter(x(arr(:, 8) == 5), arr(arr(:, 8) == 5, 5), 'yellow', '.');
scatter(x(arr(:, 8) == 3), arr(arr(:, 8) == 3, 5), 'green', '.');
scatter(x(arr(:, 8) == 2), arr(arr(:, 8) == 2, 5), 'red', '.');
scatter(x(arr(:, 8) == 1), arr(arr(:, 8) == 1, 5), 'blue', '.');
legend('Vehicle/Bus/Train','Meeting/Computer','Walk','Eat');
hold off

% %% Plotting Dataset
figure(1) % Scatter plot
x = 1:length(arr);
hold on
scatter(arr(arr(:, 8) == 2,4),  arr(arr(:, 8) == 2, 5), 'blue', '.');
scatter(arr(arr(:, 8) == 3,4), arr(arr(:, 8) == 3, 5), 'red', '.');
legend('Walk','Meeting/Computer');
title('Scatter Plot of Dataset');
xlabel('Acceleration (Z-axis)');
ylabel('Gyroscope(X-axis)');

%% obtaining X and Y axes from data
% WalkX = arr(arr(:, 8) == 2,4);
% MeetingComputerX = arr(arr(:, 8) == 3,4);
% allX = [WalkX; MeetingComputerX];
% 
% WalkY = arr(arr(:, 8) == 2, 5);
% MeetingComputerY = arr(arr(:, 8) == 3, 5);
% allY = [WalkY; MeetingComputerY];
% 
% %X and Y axes combined
% XY = [allX, allY];
% 
% WalkLabel = arr(arr(:, 8) == 2, 8);
% MeetingComputerLabel = arr(arr(:, 8) == 3,8);
% allLabel = [WalkLabel;MeetingComputerLabel];
% 
% %XY and Label combined
% XYLabel = [XY,allLabel];

%% CVPARTITION

% %Partition Data by witholding 10% as testing data
% cv = cvpartition(length(XY(:,1)),'HoldOut',0.1);
% idx = cv.test;
% % Separate into training and test data
dataTrain = [training_ACC_Z(:,1),training_GYRO_X(:,1)];
dataTest  = [testing_ACC_Z(:,1),testing_GYRO_X(:,1)];
dataTrainLabels=[training_GYRO_X(:,2)];
dataTestLabels = [testing_GYRO_X(:,2)];
% %Partition Data by witholding 10% as testing data
% cvSVM = cvpartition(length(XYLabel(:,1)),'HoldOut',0.1);
% idxSVM = cvSVM.test;
% % Separate into training and test data
% dataTrainSVM = XYLabel(~idxSVM,:);
% dataTestSVM = XYLabel(idxSVM,:);
%% MAP Classification
%fitting gaussian mixture model into data
gmm = fitgmdist(dataTrain ,2,'RegularizationValue',.01);

%construct clusters from gaussian 
[idxTrain,nlogL,P,logpdf,d2] = cluster(gmm,dataTrain);

%Confusion matrix for MAP Classifier on Training Data
C = confusionmat(dataTrainLabels,idxTrain);

%calculate errors
countMAPgaussiantrainerror = 0;
for e =1:length(dataTrain(:,1))
    if idxTrain(e)~=dataTrainLabels(e)  
        countMAPgaussiantrainerror = countMAPgaussiantrainerror +1;
        indexMAPTrain(countMAPgaussiantrainerror) = e;
    end
end
countMAPgaussiantrainerror = countMAPgaussiantrainerror/length(training_GYRO_X);

for m = 1:length(indexMAPTrain)
    mislabeledMAPTrain(m,: ) = dataTrain(indexMAPTrain(m),:);
end

%plot 
figure(2)
scatter(mislabeledMAPTrain(:,1),mislabeledMAPTrain(:,2),'k');
hold on
gscatter(dataTrain(:,1), dataTrain(:,2), idxTrain,'rb','.x' );
legend( '= Misclassified', '= Meeting/Using Computer','= Walking');
title('MAP Classifier on Training Data','FontSize',20);
xlabel('Acceleration in Z-axis','FontSize', 20);
ylabel('Gyroscope in X-axis', 'FontSize', 20);

%% MAP Classifier on Testing Data
[idxTest, nlogLTest, PTest,logpdfTest,d2Test] = cluster(gmm,dataTest);

%calculating Error 
countMAPgaussiantesterror = 0;
for e =1:length(dataTest(:,1))
    if idxTest(e)~=dataTestLabels(e)
        countMAPgaussiantesterror = countMAPgaussiantesterror +1;
        indexMAPTest(countMAPgaussiantesterror) = e;
    end
end
countMAPgaussiantesterror = countMAPgaussiantesterror/length(testing_GYRO_X);

for m = 1:length(indexMAPTest)
    mislabeledMAPTest(m,: ) = dataTest(indexMAPTest(m),:);
end

%plot
figure(3) 
scatter(mislabeledMAPTest(:,1),mislabeledMAPTest(:,2),'k');
hold on
gscatter(dataTest(:,1),dataTest(:,2),idxTest,'rb','.x'  );
legend( '= Misclassfied','= Meeting/Using Computer','= Walking')
title('MAP Classifier on Testing Data','FontSize',20);
xlabel('Acceleration in Z-axis','FontSize',20 );
ylabel('Gyroscope in X-axis','FontSize',20 );

%Confusion Matrix on Testing Data
C = confusionmat(dataTestLabels,idxTest);

%% Gaussian SVM CLassification

%Finding hyperparameters for Gaussian SVM

%Mdl = fitcsvm(dataTrain,dataTrainLabels,'KernelFunction','gaussian','OptimizeHyperparameters', 'auto');

%Fitting Gaussian SVM to training data with optimized hyperparameters
gaussianSVMtrained = fitcsvm(dataTrain,dataTrainLabels,'BoxConstraint', 1.1602 ,'KernelFunction','gaussian','KernelScale',936.67);
Cgaussian = crossval(gaussianSVMtrained,'KFold',10);
klossgaussian = kfoldLoss(Cgaussian);

%predicting labels for training data with fitted Gaussian SVM
gaussianfit = predict(gaussianSVMtrained,dataTrain);

%finding loss value 
gaussianlosstraining = loss(gaussianSVMtrained,dataTrain,gaussianfit);

%Calculating Error between Gaussian SVM and Training Data Labels
countgaussiantrainerror = 0;
for e =1:length(dataTrain(:,1))
    if gaussianfit(e) ~=dataTrainLabels(e)
        countgaussiantrainerror = countgaussiantrainerror +1;
        indexGAUSSIANTrain(countgaussiantrainerror)=e;
    end
end
countgaussiantrainerror = countgaussiantrainerror/length(training_GYRO_X);

for m = 1:length(indexGAUSSIANTrain)
    mislabeledGAUSSIANTrain(m,: ) = dataTrain(indexGAUSSIANTrain(m),:);
end

%Plotting Classification Results of Gaussian SVM on Training Data
figure(4)
scatter(mislabeledGAUSSIANTrain(:,1),mislabeledGAUSSIANTrain(:,2),'k');
hold on
gscatter(dataTrain(:,1),dataTrain(:,2),gaussianfit,'rb','.x');
legend('= Misclassified','= Meeting/Using Computer','= Walking')
title('Gaussian SVM on Training Data','FontSize',20);
xlabel('Acceleration in Z-axis','FontSize',20);
ylabel('Gyroscope in X-axis','FontSize',20);

%Confusion Matrix for Gaussian SVM Classifier on Training Data
C = confusionmat(dataTrainLabels,gaussianfit);
%% Gaussian SVM Testing 
testgaussianfit = predict(gaussianSVMtrained,dataTest);
gaussianlosstest= loss(gaussianSVMtrained,dataTest,testgaussianfit);

%calculating errors
countgaussiantesterror = 0;
for e =1:length(dataTest(:,1))
    if testgaussianfit(e) ~=dataTestLabels(e)
        countgaussiantesterror = countgaussiantesterror +1;
        indexGAUSSIANTest(countgaussiantesterror)=e;
    end
end
countgaussiantesterror = countgaussiantesterror/length(testing_GYRO_X);

for m = 1:length(indexGAUSSIANTest)
    mislabeledGAUSSIANTest(m,: ) = dataTest(indexGAUSSIANTest(m),:);
end

%plot
figure(6)
scatter(mislabeledGAUSSIANTest(:,1),mislabeledGAUSSIANTest(:,2),'k');
hold on
gscatter(dataTest(:,1),dataTest(:,2),testgaussianfit,'rb','.x');
legend('= Misclassified','=  Meeting/Using Computer','= Walking')
title('Gaussian SVM on Testing Data', 'FontSize',20);
xlabel('Acceleration in Z-axis', 'FontSize',20 );
ylabel('Gyroscope in X-axis', 'FontSize',20 );

%Confusion Matrix of Gaussian Classifier on Testing Data
C = confusionmat(dataTestLabels,testgaussianfit);



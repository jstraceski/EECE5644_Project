%% MACHINE LEARNING AND PATTERN RECOGNITION - fINAL PROJECT, 2019

clf
glassesData = readtable('glasses.csv');
report = readtable('report.csv');

div = 100;

reduction = ceil(height(glassesData(:, 1))/div); % Divide data into groups of 100
arr = zeros(reduction, 9);

for i = 1:div:height(glassesData(:, 1)) % create cell for parsing each data entry
    formatIn = 'yyyy/mm/dd HH:MM:SS.FFF';
    arr(ceil(i/div), 1) = datenum(glassesData{i, 2},formatIn); % Date
    arr(ceil(i/div), 2) = glassesData{i, 3}; % Acceleration x
    arr(ceil(i/div), 3) = glassesData{i, 4}; % Acceleration y 
    arr(ceil(i/div), 4) = glassesData{i, 5}; % Acceleration z
    arr(ceil(i/div), 5) = glassesData{i, 6}; % Gyro x
    arr(ceil(i/div), 6) = glassesData{i, 7}; % Gyro y
    arr(ceil(i/div), 7) = glassesData{i, 8}; % Gyro z
    
    arr(ceil(i/div), 9) = sqrt(glassesData{i, 3}^2 + glassesData{i, 4}^2 + glassesData{i, 5}^2); % Acceleration
    arr(ceil(i/div), 10) = sqrt(glassesData{i, 6}^2 + glassesData{i, 7}^2 + glassesData{i, 8}^2); % Gyro
end

arr(:, 8) = 0; % Set the label for each class

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

% Identify data belonging to different activities
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

%% Decide what data to take
figure(1) % Acc in x
x = 1:length(arr);
hold on
%scatter(arr(arr(:, 8) == 4, 10), arr(arr(:, 8) == 4, 9), 'black', '.')
scatter(x(arr(:, 8) == 5), arr(arr(:, 8) == 5, 2), 'yellow', '.');
scatter(x(arr(:, 8) == 3), arr(arr(:, 8) == 3, 2), 'green', '.');
scatter(x(arr(:, 8) == 2), arr(arr(:, 8) == 2, 2), 'red', '.');
scatter(x(arr(:, 8) == 1), arr(arr(:, 8) == 1, 2), 'blue', '.');
legend('Vehicle/Bus/Train','Meeting/Computer','Walk','Eat');

figure(2) % Acc in y
x = 1:length(arr);
hold on
%scatter(arr(arr(:, 8) == 4, 10), arr(arr(:, 8) == 4, 9), 'black', '.')
scatter(x(arr(:, 8) == 5), arr(arr(:, 8) == 5, 3), 'yellow', '.');
scatter(x(arr(:, 8) == 3), arr(arr(:, 8) == 3, 3), 'green', '.');
scatter(x(arr(:, 8) == 2), arr(arr(:, 8) == 2, 3), 'red', '.');
scatter(x(arr(:, 8) == 1), arr(arr(:, 8) == 1, 3), 'blue', '.');
legend('Vehicle/Bus/Train','Meeting/Computer','Walk','Eat');

figure(3) % Acc in z
x = 1:length(arr);
hold on
%scatter(arr(arr(:, 8) == 4, 10), arr(arr(:, 8) == 4, 9), 'black', '.')
scatter(x(arr(:, 8) == 5), arr(arr(:, 8) == 5, 4), 'yellow', '.');
scatter(x(arr(:, 8) == 3), arr(arr(:, 8) == 3, 4), 'green', '.');
scatter(x(arr(:, 8) == 2), arr(arr(:, 8) == 2, 4), 'red', '.');
scatter(x(arr(:, 8) == 1), arr(arr(:, 8) == 1, 4), 'blue', '.');
legend('Vehicle/Bus/Train','Meeting/Computer','Walk','Eat');

figure(4) % Gyro in x
x = 1:length(arr);
hold on
%scatter(arr(arr(:, 8) == 4, 10), arr(arr(:, 8) == 4, 9), 'black', '.')
scatter(x(arr(:, 8) == 5), arr(arr(:, 8) == 5, 5), 'yellow', '.');
scatter(x(arr(:, 8) == 3), arr(arr(:, 8) == 3, 5), 'green', '.');
scatter(x(arr(:, 8) == 2), arr(arr(:, 8) == 2, 5), 'red', '.');
scatter(x(arr(:, 8) == 1), arr(arr(:, 8) == 1, 5), 'blue', '.');
legend('Vehicle/Bus/Train','Meeting/Computer','Walk','Eat');

figure(5) % Gyro in y
x = 1:length(arr);
hold on
%scatter(arr(arr(:, 8) == 4, 10), arr(arr(:, 8) == 4, 9), 'black', '.')
scatter(x(arr(:, 8) == 5), arr(arr(:, 8) == 5, 6), 'yellow', '.');
scatter(x(arr(:, 8) == 3), arr(arr(:, 8) == 3, 6), 'green', '.');
scatter(x(arr(:, 8) == 2), arr(arr(:, 8) == 2, 6), 'red', '.');
scatter(x(arr(:, 8) == 1), arr(arr(:, 8) == 1, 6), 'blue', '.');
legend('Vehicle/Bus/Train','Meeting/Computer','Walk','Eat');

figure(6) % Gyro in z
x = 1:length(arr);
hold on
scatter(x(arr(:, 8) == 4),  arr(arr(:, 8) == 4, 7), 'black', '.');
scatter(x(arr(:, 8) == 5), arr(arr(:, 8) == 5, 7), 'yellow', '.');
scatter(x(arr(:, 8) == 3), arr(arr(:, 8) == 3, 7), 'green', '.');
scatter(x(arr(:, 8) == 2), arr(arr(:, 8) == 2, 7), 'red', '.');
scatter(x(arr(:, 8) == 1), arr(arr(:, 8) == 1, 7), 'blue', '.');
legend('Vehicle/Bus/Train','Meeting/Computer','Walk','Eat');

% Selected data: acceleration in z-axis and gyro in x axis as the data
% containing relevant information 
figure(9) % Scatter plot
x = 1:length(arr);
hold on
scatter(arr(arr(:, 8) == 2, 4),  arr(arr(:, 8) == 2, 5), 'blue', 'x');
scatter(arr(arr(:, 8) == 3, 4), arr(arr(:, 8) == 3, 5), 'red', '.');
legend();
xlabel(['Acceleration in z axis'], 'Interpreter','latex','Fontsize',12);
ylabel(['Rotation in x axis (pitch)'], 'Interpreter','latex','Fontsize',12);
grid minor
set(gcf,'color','w');
legend ({'Walking','Meeting/Using Computer'},'Fontsize',12,'Interpreter','latex','location','Northwest');
title('Scatter plot of the data')

% Create testing and training datasets
total_samples = size(arr(arr(:, 8) == 3, 4),1)+size(arr(arr(:, 8) == 2, 5),1);
test_samples = round(0.1*total_samples);
indexes= randperm(total_samples,total_samples);
index_test = indexes(1:round(0.1*total_samples));
index_train = indexes(round(0.1*total_samples):end);

data_x = [arr(arr(:, 8) == 3, 4) ; arr(arr(:, 8) == 2, 4)];
labels_x = [ones(size(arr(arr(:, 8) == 3, 4),1),1);2*ones(size(arr(arr(:, 8) == 2, 4),1),1)];
acce_z = [data_x,labels_x];

data_xgy = [arr(arr(:, 8) == 3, 5) ; arr(arr(:, 8) == 2, 5)];
labels_xgy = [ones(size(arr(arr(:, 8) == 3, 4),1),1);2*ones(size(arr(arr(:, 8) == 2, 4),1),1)];
gyro_x = [data_xgy,labels_xgy];

training_ACC_Z = acce_z(index_train,:);
training_GYRO_X = gyro_x(index_train,:);
testing_ACC_Z = acce_z(index_test,:);
testing_GYRO_X = gyro_x(index_test,:);

% We save the dataset to use the same between different algorithms
% save('training_ACC_Z.mat','training_ACC_Z');
% save('training_GYRO_X.mat','training_GYRO_X');
% save('testing_ACC_Z.mat','testing_ACC_Z');
% save('testing_GYRO_X.mat','testing_GYRO_X');

load('training_ACC_Z.mat','training_ACC_Z');
load('training_GYRO_X.mat','training_GYRO_X');
load('testing_ACC_Z.mat','testing_ACC_Z');
load('testing_GYRO_X.mat','testing_GYRO_X');

figure(10) % Scatter plot for testing data
x = 1:length(arr);
hold on
scatter(arr(arr(:, 8) == 2, 4),  arr(arr(:, 8) == 2, 5), 'blue', 'x');
scatter(arr(arr(:, 8) == 3, 4), arr(arr(:, 8) == 3, 5), 'red', '.');
scatter(acce_z(index_test),gyro_x(index_test),'k');
legend();
xlabel(['Acceleration in z axis'], 'Interpreter','latex','Fontsize',12);
ylabel(['Rotation in x axis (pitch)'], 'Interpreter','latex','Fontsize',12);
grid minor
set(gcf,'color','w');
legend ({'Walk','Meeting/Computer','testing data'},'Fontsize',12,'Interpreter','latex','location','Northwest');
title('Scatter plot with test dataset')


%% SVM LINEAR CLASSIFIER
% Testing and training datasets
training_set = [training_ACC_Z(:,1),training_GYRO_X(:,1)];
true_labels = training_ACC_Z(:,2);
testing_set = [testing_ACC_Z(:,1),testing_GYRO_X(:,1)];

% Linear SVM for training
mdlLIN = fitcsvm(training_set, true_labels,'Standardize',true,'KernelFunction','linear','KernelScale','auto');
NEWlabelLIN = predict(mdlLIN,training_set);

% Confusion matrix and error calculation 
errorLIN = 0;
errorWalk = 0;
errorMeeting = 0;
guessWalk = 0;
guessMeeting = 0;

% Probability of error calculation
for i = 1:size(NEWlabelLIN,1)
    if ((NEWlabelLIN(i)~=true_labels(i))==1)
        errorLIN = errorLIN+1;
    end
     if (((NEWlabelLIN(i)==true_labels(i))&&(NEWlabelLIN(i)==1))==1)
        guessWalk = guessWalk+1;
     end
     if (((NEWlabelLIN(i)==true_labels(i))&&(NEWlabelLIN(i)==2))==1)
        guessMeeting = guessMeeting+1;
     end
     if (((NEWlabelLIN(i)~=true_labels(i))&&(true_labels(i)==1))==1)
        errorWalk = errorWalk+1;
     end
     if (((NEWlabelLIN(i)~=true_labels(i))&&(true_labels(i)==2))==1)
        errorMeeting = errorMeeting+1;
     end
end
errorLINN = (errorLIN/size(NEWlabelLIN,1))*100; % Percentage of error

% Linear SVM for testing
true_labels_test = testing_ACC_Z(:,2);

NEWlabelLINtest = predict(mdlLIN,testing_set);

% Probability of error calculation
errorLIN = 0;
for i = 1:size(NEWlabelLINtest,1)
    if ((NEWlabelLINtest(i)~=true_labels_test(i))==1)
        errorLIN = errorLIN+1;
    end
end
errorLINtest = (errorLIN/size(NEWlabelLINtest,1))*100; % Percentage of error

%%%
% Confusion matrix and error calculation 
errorLIN = 0;
errorWalk = 0;
errorMeeting = 0;
guessWalk = 0;
guessMeeting = 0;
for i = 1:size(NEWlabelLINtest,1)
    if ((NEWlabelLINtest(i)~=true_labels_test(i))==1)
        errorLIN = errorLIN+1;
    end
     if (((NEWlabelLINtest(i)==true_labels_test(i))&&(NEWlabelLINtest(i)==1))==1)
        guessWalk = guessWalk+1;
     end
     if (((NEWlabelLINtest(i)==true_labels_test(i))&&(NEWlabelLINtest(i)==2))==1)
        guessMeeting = guessMeeting+1;
     end
     if (((NEWlabelLINtest(i)~=true_labels_test(i))&&(true_labels_test(i)==1))==1)
        errorWalk = errorWalk+1;
     end
     if (((NEWlabelLINtest(i)~=true_labels_test(i))&&(true_labels_test(i)==2))==1)
        errorMeeting = errorMeeting+1;
     end
end
%%%


% Plot for linear case, training
walkingLin = [];
computerLin=[];
errorsLin = [];

for i = 1:size(NEWlabelLIN,1)
    if ((NEWlabelLIN(i)==1)==1)
          computerLin = [computerLin; training_set(i,:)];
    else
        walkingLin = [walkingLin; training_set(i,:)];
    end
        if ((NEWlabelLIN(i)~=true_labels(i))==1)
        errorsLin = [errorsLin; training_set(i,:) ];
    end
end
figure(11) % Scatter plot
x = 1:length(arr);
hold on
scatter(errorsLin(:,1),  errorsLin(:,2),'k');
scatter(walkingLin(:,1),  walkingLin(:,2), 'blue', 'x');
scatter(computerLin(:,1),  computerLin(:,2), 'red', '.');
legend();
xlabel(['Acceleration in z axis'], 'Interpreter','latex','Fontsize',12);
ylabel(['Rotation in x axis (pitch)'], 'Interpreter','latex','Fontsize',12);
grid minor
set(gcf,'color','w');
legend ({'Errors','Walking','Meeting/Using Computer'},'Fontsize',12,'Interpreter','latex','location','Northwest');
title('Linear SVM, Training Dataset')


% Plot for linear case, testing
walkingLintest = [];
computerLintest=[];
errorsLintest = [];

for i = 1:size(NEWlabelLINtest,1)
    if ((NEWlabelLINtest(i)==1)==1)
      computerLintest = [computerLintest; testing_set(i,:)];
    else
         walkingLintest = [walkingLintest; testing_set(i,:)];
        
    end
        if ((NEWlabelLINtest(i)~=true_labels_test(i))==1)
        errorsLintest = [errorsLintest; testing_set(i,:) ];
    end
end
figure(12) % Scatter plot
x = 1:length(arr);
hold on
scatter(errorsLintest(:,1),  errorsLintest(:,2),'k');
scatter(walkingLintest(:,1),  walkingLintest(:,2), 'blue', 'x');
scatter(computerLintest(:,1),  computerLintest(:,2), 'red', '.');
legend();
xlabel(['Acceleration in z axis'], 'Interpreter','latex','Fontsize',12);
ylabel(['Rotation in x axis (pitch)'], 'Interpreter','latex','Fontsize',12);
grid minor
set(gcf,'color','w');
legend ({'Errors','Walking','Meeting/Using Computer'},'Fontsize',12,'Interpreter','latex','location','Northwest');
title('Linear SVM, Testing Dataset')
grid minor



%% SVM GAUSSIAN CLASSIFIER
% Testing and training datasets
training_set = [training_ACC_Z(:,1),training_GYRO_X(:,1)];
true_labels = training_ACC_Z(:,2);
testing_set = [testing_ACC_Z(:,1),testing_GYRO_X(:,1)];

% Gaussian SVM for training

mdlGAUSS = fitcsvm(training_set, true_labels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
NEWlabelGAUSS = predict(mdlGAUSS,training_set);

% Probability of error calculation
errorGAUSS = 0;
for i = 1:size(NEWlabelGAUSS,1)
    if ((NEWlabelGAUSS(i)~=true_labels(i))==1)
        errorGAUSS = errorGAUSS+1;
    end
end
errorGAUSS = (errorGAUSS/size(NEWlabelGAUSS,1))*100; % Percentage of error

% Gaussian SVM for testing
true_labels_test = testing_ACC_Z(:,2);

NEWlabelGAUSS = predict(mdlGAUSS,testing_set);

% Probability of error calculation
errorGAUS = 0;
for i = 1:size(NEWlabelGAUSS,1)
    if ((NEWlabelGAUSS(i)~=true_labels_test(i))==1)
        errorGAUS = errorGAUS+1;
    end
end
errorGAUStest = (errorGAUS/size(NEWlabelGAUSS,1))*100; % Percentage of error

computer=[];
walk=[];

%% SVM GAUSSIAN CLASSIFIER. THE KERNEL TRICK

% Add third dimension to data
for i = 1:size(training_ACC_Z,1)
    if(training_ACC_Z(i,2)==1)
        training_ACC_Z(i,3)=20;
        training_GYRO_X(i,3)=20;
        computer = [computer; training_ACC_Z(i,1),training_GYRO_X(i,1),training_ACC_Z(i,3)];
    else 
        training_ACC_Z(i,3)=0;
        training_GYRO_X(i,3)=0;
        walk = [walk; training_ACC_Z(i,1),training_GYRO_X(i,1),training_ACC_Z(i,3)];
    end
end

for i = 1:size(testing_ACC_Z,1)
    if(testing_ACC_Z(i,2)==1)
        testing_ACC_Z(i,3)=20;
        testing_GYRO_X(i,3)=20;
    else 
        testing_ACC_Z(i,3)=0;
        testing_GYRO_X(i,3)=0;
    end
end

% Testing and training datasets
training_set = [training_ACC_Z(:,1),training_GYRO_X(:,1),training_GYRO_X(:,3)];
true_labels = training_ACC_Z(:,2);
testing_set = [testing_ACC_Z(:,1),testing_GYRO_X(:,1),testing_GYRO_X(:,3)];

% Representation of higher dimensional data
figure(13)
scatter3(walk(:,1),walk(:,2),walk(:,3), 'blue', 'x');
hold on
scatter3(computer(:,1),computer(:,2),computer(:,3), 'red', '.');
xlabel(['Acceleration in z axis'], 'Interpreter','latex','Fontsize',14);
ylabel(['Rotation in x axis (pitch)'], 'Interpreter','latex','Fontsize',14);
grid minor
set(gcf,'color','w');
legend ({'Walking','Meeting/Using Computer'},'Fontsize',18,'Interpreter','latex','location','Northwest');
grid minor


% Call to Gaussian SVM with kernel trick (3D data)

mdlGAUSS = fitcsvm(training_set, true_labels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
NEWlabelGAUSS = predict(mdlGAUSS,training_set);

% probability of error calculation
errorGAUSS = 0;
for i = 1:size(NEWlabelGAUSS,1)
    if ((NEWlabelGAUSS(i)~=true_labels(i))==1)
        errorGAUSS = errorGAUSS+1;
    end
end
errorGAUSS = (errorGAUSS/size(NEWlabelGAUSS,1))*100; % Percentage of error

%%%%
% Confusion matrix and error calculation 
errorLIN = 0;
errorWalk = 0;
errorMeeting = 0;
guessWalk = 0;
guessMeeting = 0;
for i = 1:size(NEWlabelGAUSS,1)
    if ((NEWlabelGAUSS(i)~=true_labels(i))==1)
        errorLIN = errorLIN+1;
    end
     if (((NEWlabelGAUSS(i)==true_labels(i))&&(NEWlabelGAUSS(i)==1))==1)
        guessWalk = guessWalk+1;
     end
     if (((NEWlabelGAUSS(i)==true_labels(i))&&(NEWlabelGAUSS(i)==2))==1)
        guessMeeting = guessMeeting+1;
     end
     if (((NEWlabelGAUSS(i)~=true_labels(i))&&(true_labels(i)==1))==1)
        errorWalk = errorWalk+1;
     end
     if (((NEWlabelGAUSS(i)~=true_labels(i))&&(true_labels(i)==2))==1)
        errorMeeting = errorMeeting+1;
     end
end

%%%

% Plot for gauss case kernel, training
walkingGaus = [];
computerGaus=[];
errorGAUSS = [];

for i = 1:size(NEWlabelGAUSS,1)
    if ((NEWlabelGAUSS(i)==1)==1)
          computerGaus = [computerGaus; training_set(i,:)];
    else
        walkingGaus = [walkingGaus; training_set(i,:)];
    end
        if ((NEWlabelGAUSS(i)~=true_labels(i))==1)
        errorGAUSS = [errorGAUSS; training_set(i,:) ];
    end
end
figure(14) % Scatter plot
x = 1:length(arr);
hold on
%scatter(errorGAUSS(:,1),  errorGAUSS(:,2),'k');
scatter(walkingGaus(:,1),  walkingGaus(:,2), 'blue', 'x');
scatter(computerGaus(:,1),  computerGaus(:,2), 'red', '.');
legend();
xlabel(['Acceleration in z axis'], 'Interpreter','latex','Fontsize',12);
ylabel(['Rotation in x axis (pitch)'], 'Interpreter','latex','Fontsize',12);
grid minor
set(gcf,'color','w');
legend ({'Walking','Meeting/Using Computer'},'Fontsize',12,'Interpreter','latex','location','Northwest');
title('Kernel Trick Gaussian SVM, Training Dataset')

%%%
% Testing data
true_labels_test = testing_ACC_Z(:,2);

NEWlabelGAUSS = predict(mdlGAUSS,testing_set);

% probability of error calculation
errorGAUS = 0;
for i = 1:size(NEWlabelGAUSS,1)
    if ((NEWlabelGAUSS(i)~=true_labels_test(i))==1)
        errorGAUS = errorGAUS+1;
    end
end

errorGAUStest = (errorGAUS/size(NEWlabelGAUSS,1))*100; % Percentage of error

%%%%
% Confusion matrix and error calculation 
errorLIN = 0;
errorWalk = 0;
errorMeeting = 0;
guessWalk = 0;
guessMeeting = 0;
for i = 1:size(NEWlabelGAUSS,1)
    if ((NEWlabelGAUSS(i)~=true_labels_test(i))==1)
        errorLIN = errorLIN+1;
    end
     if (((NEWlabelGAUSS(i)==true_labels_test(i))&&(NEWlabelGAUSS(i)==1))==1)
        guessWalk = guessWalk+1;
     end
     if (((NEWlabelGAUSS(i)==true_labels_test(i))&&(NEWlabelGAUSS(i)==2))==1)
        guessMeeting = guessMeeting+1;
     end
     if (((NEWlabelGAUSS(i)~=true_labels_test(i))&&(true_labels_test(i)==1))==1)
        errorWalk = errorWalk+1;
     end
     if (((NEWlabelGAUSS(i)~=true_labels_test(i))&&(true_labels_test(i)==2))==1)
        errorMeeting = errorMeeting+1;
     end
end



% Plot for linear case, testing
walkingLintest = [];
computerLintest=[];
errorsLintest = [];

for i = 1:size(NEWlabelGAUSS,1)
    if ((NEWlabelGAUSS(i)==1)==1)
      computerLintest = [computerLintest; testing_set(i,:)];
    else
         walkingLintest = [walkingLintest; testing_set(i,:)];
        
    end
        if ((NEWlabelGAUSS(i)~=true_labels_test(i))==1)
        errorsLintest = [errorsLintest; testing_set(i,:) ];
    end
end
figure(15) % Scatter plot
x = 1:length(arr);
hold on
scatter(errorsLintest(:,1),  errorsLintest(:,2),'k');
scatter(walkingLintest(:,1),  walkingLintest(:,2), 'blue', 'x');
scatter(computerLintest(:,1),  computerLintest(:,2), 'red', '.');
legend();
xlabel(['Acceleration in z axis'], 'Interpreter','latex','Fontsize',20);
ylabel(['Rotation in x axis (pitch)'], 'Interpreter','latex','Fontsize',20);
grid minor
set(gcf,'color','w');
legend ({'Errors','Walking','Meeting/Using Computer'},'Fontsize',16,'Interpreter','latex','location','Northwest');
 title('Kernel Trick Gaussian SVM, Testing Dataset')
grid minor



%% PCA (Principal component analysis)
format long

training_ACC_Z = acce_z(index_train,:);
training_GYRO_X = gyro_x(index_train,:);
testing_ACC_Z = acce_z(index_test,:);
testing_GYRO_X = gyro_x(index_test,:);

data_x = [training_ACC_Z(:,1),training_GYRO_X(:,1)]';
corr_data = data_x*data_x'; 
[V,D] = eig(corr_data);
eigenval = diag(D);
eigenvalcart =eigenval./ max(abs(eigenval));
eigenvalcart= sort(eigenvalcart,'descend');

 [TH,R] = cart2pol(training_ACC_Z(:,1),training_GYRO_X(:,1));
data_xx = [TH,R]';
corr_dataxx = data_xx*data_xx'; 
[V,D] = eig(corr_dataxx);
eigenval = diag(D);
eigenvalpol =eigenval./ max(abs(eigenval));
eigenvalpol= sort(eigenvalpol,'descend');

figure(16)
plot(eigenvalcart,'o-r')
hold on
plot(eigenvalpol,'o-b')
xlabel(['Dimension'], 'Interpreter','latex','Fontsize',20);
ylabel(['Normalized eigenvalue'], 'Interpreter','latex','Fontsize',20);
grid minor
set(gcf,'color','w');
legend ({'Cartesian Coordinates','Polar Coordinates'},'Fontsize',16,'Interpreter','latex','location','Northwest');
 title('Normalized Eigenvalues for Different Coordinate Systems')
grid minor

%% 
coeff1 = pca([training_ACC_Z(:,1),training_GYRO_X(:,1)]);
coeff2 = pca([R,TH]);


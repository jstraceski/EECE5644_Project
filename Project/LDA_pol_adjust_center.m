% LDA classification using Accel_Z and Gyro_X converted to polar values
% center adjusted to average of class centroids
format compact

load('training_ACC_Z.mat');
load('training_GYRO_X.mat');
load('testing_ACC_Z.mat');
load('testing_GYRO_X.mat');

training_dat = [training_ACC_Z(:,1) training_GYRO_X];
testing_dat = [testing_ACC_Z(:,1) testing_GYRO_X];

walk_dat = training_dat(training_dat(:,3)==1,1:2);
meet_dat = training_dat(training_dat(:,3)==2,1:2);

%---------------Adjust to Average of Class Centroids---------------%

A_pol = training_dat(:,3);
X_dat = training_dat(:,1:2);

class1_adj = X_dat(A_pol==1,:);
class2_adj = X_dat(A_pol==2,:);

[~, Mu1_adj] = robustcov(class1_adj);
[~, Mu2_adj] = robustcov(class2_adj);

mu_adj = [mean([Mu1_adj(1); Mu1_adj(1)]) mean([Mu1_adj(2); Mu1_adj(2)])];

%X_cart_cl1 = [(X_dat(A_pol==1,1) - Mu1_adj(1)) (X_dat(A_pol==1,2) - Mu1_adj(2))];
%X_cart_cl2 = [(X_dat(A_pol==2,1) - Mu2_adj(1)) (X_dat(A_pol==2,2) - Mu2_adj(2))];

%A_pol = [ones(length(X_cart_cl1),1); 2*ones(length(X_cart_cl2),1)];

X_cart = [(X_dat(:,1) - mu_adj(1)) (X_dat(:,2) - mu_adj(2))];

test_X_dat = testing_dat(:,1:2);

test_X_cart = [(test_X_dat(:,1) - mu_adj(1)) (test_X_dat(:,2) - mu_adj(2))];

%---------------Cart2Pol Conversion---------------%

[theta_train, rho_train] = cart2pol(X_cart(:,1), X_cart(:,2));
X_pol = [rho_train theta_train];

[theta_test, rho_test] = cart2pol(test_X_cart(:,1), test_X_cart(:,2));
test_X_pol = [rho_test theta_test];

%---------------LDA POLAR---------------%

class1 = X_pol(A_pol==1,:);
class2 = X_pol(A_pol==2,:);

[S1, Mu1] = robustcov(class1);
[S2, Mu2] = robustcov(class2);

Sw = S1 + S2;
SB = (Mu1 - Mu2)*(Mu1 - Mu2)';

invSw = inv(Sw);
invSw_by_SB = invSw * SB;

[V,D] = eig(invSw_by_SB);

W = V(:,1);

disc_x = W(1);
disc_y = W(2);

fd = fitcdiscr(X_pol, A_pol);
pr_vals = predict(fd, X_pol);

% W'[x1; x2] + B<0
B = fd.Coeffs(2,1).Const;  
W = fd.Coeffs(2,1).Linear;
disc_bound = @(x) B + W'*x'; % fix x to proper format
disc_bound_graph = @(x,y) B + W(1)*x + W(2)*y;
res = disc_bound(X_pol);
FLDA_training_predicted_vals = zeros(length(res), 1);
for i = 1:length(res)
    if res(i) < 0
        FLDA_training_predicted_vals(i) = 1;
    else
        FLDA_training_predicted_vals(i) = 2;
    end
end

%-------------------------Cross Val-------------------------%

test_result = disc_bound(test_X_pol);
FLDA_test_predicted_vals = zeros(length(test_result), 1);
for i = 1:length(test_result)
    if test_result(i) < 0
        FLDA_test_predicted_vals(i) = 1;
    else
        FLDA_test_predicted_vals(i) = 2;
    end
end

%-------------------------Error Calcs-------------------------%

%(observed,predicted)
train_cm = confusionmat(training_dat(:,3), FLDA_training_predicted_vals);
test_cm = confusionmat(testing_dat(:,3), FLDA_test_predicted_vals);

% (False_Positives + True_Negatives) / Total_Samples = error
train_err = (train_cm(1,2)+ train_cm(2,1)) / sum(sum(train_cm))
test_err = (test_cm(1,2)+ test_cm(2,1)) / sum(sum(test_cm))

%-------------------------LDA Graph-------------------------%

figure()
set(gcf,'color','w');
title("Fisher's LDA with Polar-Adjusted Values");
hold on
boundary_line = fimplicit(disc_bound_graph, [min(X_pol(:,1)) max(X_pol(:,1)) min(X_pol(:,2)) max(X_pol(:,2))]);
%gscatter(X_pol(:,1), X_pol(:,2), A_pol,'rc', '..');

gscatter(X_pol(:,1),...
    X_pol(:,2),...
    FLDA_training_predicted_vals,'rb', '.x');

scatter(X_pol((A_pol~=FLDA_training_predicted_vals),1),...
    X_pol((A_pol~=FLDA_training_predicted_vals),2), 20, 'ko');

plot(Mu1(1), Mu1(2), 'b+', 'MarkerSize', 10, 'LineWidth', 2);
plot(Mu2(1), Mu2(2), 'm+', 'MarkerSize', 10, 'LineWidth', 2);
lgd = legend('Decision Boundary', 'Meeting/Computer', 'Walking', 'Misclassified',...
    'Meeting/Computer Centroid', 'Walking Centroid');
set(lgd,'FontSize',10);
hold off
legend('show')
xlabel('Rho for Acceleration(Z)/Gyroscopic(X) points');
ylabel('Theta for Acceleration(Z)/Gyroscopic(X) points');

%{
figure()
title("Fisher's LDA Predicted with Polar Converted Values Adjusted for Averaged Class Centroids");
hold on
boundary_line_2 = fimplicit(disc_bound_graph, [min(X_pol(:,1)) max(X_pol(:,1)) min(X_pol(:,2)) max(X_pol(:,2))]);
gscatter(X_pol(:,1), X_pol(:,2), FLDA_training_predicted_vals,'rb', '..');
legend('Decision Boundary','Meeting/Computer', 'Walking');
hold off
legend('show')
xlabel('Rho for Acceleration(Z)/Gyroscopic(X) points');
ylabel('Theta for Acceleration(Z)/Gyroscopic(X) points');
%}

%-------------------------Projection Graph-------------------------%

figure()

y1_w = class1*W;
y2_w = class2*W;

minY = min([min(y1_w),min(y2_w)]);
maxY = max([max(y1_w),max(y2_w)]);
y_w = minY:0.05:maxY;

y1_w_mu = mean(y1_w);
y2_w_mu = mean(y2_w);
y1_w_sig = std(y1_w);
y2_w_sig = std(y2_w);

y1_w_pdf = mvnpdf(y_w', y1_w_mu, y1_w_sig);
y2_w_pdf = mvnpdf(y_w', y2_w_mu, y2_w_sig);

hold on
plot(y_w, y1_w_pdf, 'c-');
plot(y_w, y2_w_pdf, 'r-');
legend('Meeting/Computer', 'Walking');

%-------------------------TEST DATA-----------------------------%

test_A_pol = testing_dat(:,3);

figure()
set(gcf,'color','w');
title("Test Data Fisher's LDA with Polar-Adjusted Values");
hold on
boundary_line = fimplicit(disc_bound_graph,...
    [min(test_X_pol(:,1)) max(test_X_pol(:,1)) min(test_X_pol(:,2)) max(test_X_pol(:,2))]);

gscatter(test_X_pol(:,1),...
    test_X_pol(:,2),...
    FLDA_test_predicted_vals,'rb', '.x', [5 5]);

scatter(test_X_pol((test_A_pol~=FLDA_test_predicted_vals),1),...
    test_X_pol((test_A_pol~=FLDA_test_predicted_vals),2), 20, 'ko');

plot(Mu1(1), Mu1(2), 'b+', 'MarkerSize', 10, 'LineWidth', 2);
plot(Mu2(1), Mu2(2), 'm+', 'MarkerSize', 10, 'LineWidth', 2);
lgd_2 = legend('Decision Boundary', 'Meeting/Computer', 'Walking', 'Misclassified',...
    'Meeting/Computer Centroid', 'Walking Centroid');
set(lgd_2,'FontSize',10);
hold off
legend('show')
xlabel('Rho for Acceleration(Z)/Gyroscopic(X) points');
ylabel('Theta for Acceleration(Z)/Gyroscopic(X) points');
% LDA classification using Accel_Z and Gyro_X as cartesian values
format compact

load('training_ACC_Z.mat');
load('training_GYRO_X.mat');
load('testing_ACC_Z.mat');
load('testing_GYRO_X.mat');

training_dat = [training_ACC_Z(:,1) training_GYRO_X];
testing_dat = [testing_ACC_Z(:,1) testing_GYRO_X];

walk_dat = training_dat(training_dat(:,3)==1,1:2);
meet_dat = training_dat(training_dat(:,3)==2,1:2);

A = training_dat(:,3);
X = training_dat(:,1:2);

test_A = testing_dat(:,3);
test_X = testing_dat(:,1:2);

%---------------LDA CARTESIAN---------------%

class1 = walk_dat;
class2 = meet_dat;

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

fd = fitcdiscr(X, A);
pr_vals = predict(fd, X);

% W'[x1; x2] + B<0
B = fd.Coeffs(2,1).Const;  
W = fd.Coeffs(2,1).Linear;
disc_bound = @(x) B + W'*x'; % fix x to proper format
disc_bound_graph = @(x,y) B + W(1)*x + W(2)*y;
res = disc_bound(X);
FLDA_training_predicted_vals = zeros(length(res), 1);
for i = 1:length(res)
    if res(i) < 0
        FLDA_training_predicted_vals(i) = 1;
    else
        FLDA_training_predicted_vals(i) = 2;
    end
end

%-------------------------Cross Val-------------------------%

test_result = disc_bound(test_X);
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
title("Fisher's LDA");
hold on
boundary_line = fimplicit(disc_bound_graph, [min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))]);
gscatter(X(:,1), X(:,2), A,'bg', '..');

plot(Mu1(1), Mu1(2), 'r+', 'MarkerSize', 10);
plot(Mu2(1), Mu2(2), 'k+', 'MarkerSize', 10);
hold off
legend('show')
%ylabel('accel');
%xlabel('gps');

figure()
title("Fisher's LDA Predicted");
hold on
boundary_line_2 = fimplicit(disc_bound_graph, [min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))]);
gscatter(X(:,1), X(:,2), FLDA_training_predicted_vals,'rk', 'xo');
hold off
legend('show')
%ylabel(y_axis);
%xlabel(x_axis);

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
plot(y_w, y1_w_pdf, 'r-');
plot(y_w, y2_w_pdf, 'b-');
legend('Class 1', 'Class 2');
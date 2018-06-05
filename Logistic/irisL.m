% Logistic Regression for Fisher Iris Flower Data

clear;
clc;
close all;

fprintf("Loading Data\n");
load('fisheriris.mat');

Y = species;
X = meas;
val = zeros(size(X,1),1);
val(1:50) = 1;
val(51:100) = 2;
val(101:150) = 3;

y = zeros(120,1);
y(1:40) = 1;
y(41:80) = 2;
y(81:120) = 3;
X_cv = [X(41:50,:);X(91:100,:);X(141:150,:)];
y_cv = [ones(10,1);2*ones(10,1);3*ones(10,1)];
X = [X(1:40,:);X(51:90,:);X(101:140,:)];

fprintf("Data is Loaded\n");
fprintf("Press Enter to Continue\n");
pause;

num_labels = 3;

m = size(X,1);
n = size(X,2);
lambda = 0;

fprintf("Test Case for Cost Function\n");
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10]; %reshape created a 5X3 seperate matrix and 1 to 15 is added in it.
y_t = ([1;0;1;0;1] >= 0.5); % True-False MATRIX
lambda_t = 3;
[J grad] = costFunction(theta_t, X_t, y_t, lambda_t);
fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf("Training Logistic Regression\n");
Theta = class(X,y,num_labels,lambda);
fprintf("Trained! Press Enter to continue\n");
pause;

% Accuracy
pred = predict(X,Theta);
fprintf("Accuracy is %f\n",mean(double(pred==y)) * 100);

%Cross Validation set
fprintf("Working on Cross validation set\n");
h_cv = predict(X_cv,Theta);
fprintf("Prediction of cross Validation Set is %f\n",mean(double(h_cv==y_cv)) * 100);

fprintf("Press Enter to Exit\n");
pause;

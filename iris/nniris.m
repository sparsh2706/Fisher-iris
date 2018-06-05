% Neural Networks for Iris Flower Data-Set
clear;
close all;
clc;
fprintf("Loading Data\n");
load('fisheriris.mat');

% species and meas have been loaded

fprintf("Data is Loaded\n");
Y = species;
X = meas;
y = zeros(size(X,1),1);
y(1:50) = 1;
y(51:100) = 2;
y(101:150) = 3;

fprintf("Press Enter to continue\n");
pause;

% Considering One Hidden layer with 5 units and a Output Layer with 3 units

m = size(X,1);

% Adding Bias units


input_layer_size = 4;
output_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;

theta1 = randTheta(hidden_layer_size,(input_layer_size+1));
theta2 = randTheta(output_layer_size,hidden_layer_size+1);
initial_params = [theta1(:);theta2(:)];

fprintf("Just a Check\n");
[J Dvec] = cost(initial_params,input_layer_size,hidden_layer_size,output_layer_size,X,y,0);
fprintf("J = %f\n",J);
fprintf("Press Enter to continue");
pause;

% Forward
fprintf("Training Networks\n");

% y = R^3 for multi-class

options = optimset('MaxIter',50);
lambda = 0;
costFunc = @(p) cost(p,input_layer_size,hidden_layer_size,num_labels, X, y, lambda);
[params, costJ] = fmincg(costFunc,initial_params,options);

theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

[J Dvec] = cost(params,input_layer_size,hidden_layer_size,output_layer_size,X,y,lambda);

fprintf("Neural Network Training done!\n");
fprintf("Press Enter to proceed\n");
pause;

fprintf("Error Analysis\n");
pred = predict(theta1,theta2,X);

fprintf("\nTraining Set Accuracy :%f\n",mean(double(pred == y)) * 100);

fprintf("Done. Press Enter to Exit\n");
pause;


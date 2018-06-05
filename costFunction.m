function [J grad] = costFunction(theta,X,y,lambda)

%The y here is a boolean matrix because of one vs All/ If one flower is of species 1 then y = 1 for that while for the others we have y = 0 
%Theta we get is initial_theta for fmincg

J = 0;
m = size(X,1);
n = size(X,2);
grad = zeros(size(theta));

h = sigmoid(X * theta);

% Computing Cost Function
J = (-transpose(y))*(log(h)) - (transpose(1-y))*(log(1-h));
J = J * (1/m);
thetaS = theta.^2;
J = J + (lambda/(2*m)) * (sum(thetaS) - thetaS(1,1));

% Gradient Descent For Theta

grad = (1/m)*transpose(X)*(h-y) + (lambda/m)*theta;
grad(1,1) = (1/m)*ones(1,length(y))*(h-y);

end

function [all_theta] = class(X,y,num_labels,lambda)

m = size(X,1);
n = size(X,2);
all_theta = zeros(num_labels,n+1);

X = [ones(m,1) X];

for c = 1:num_labels,
	initial_theta = zeros(n+1,1);
	options = optimset('MaxIter',18);
	[theta] = fmincg(@(t)(costFunction(t, X, (y == c), lambda)),initial_theta, options);
	all_theta(c,:) = theta;
end;

end
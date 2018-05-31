function p = predict(theta1,theta2,X)

m = size(X,1);

p = zeros(m,1);

h1 = sigmoid([ones(m,1) X] * transpose(theta1));
h2 = sigmoid([ones(m,1) h1] * transpose(theta2));
[~,p] = max(h2,[],2);

end
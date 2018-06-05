function p = predict(X,Theta)

m = size(X,1);

X = [ones(m,1) X];

h = sigmoid((Theta) * transpose(X));
h = transpose(h);

[~,p] = max(h,[],2);

end
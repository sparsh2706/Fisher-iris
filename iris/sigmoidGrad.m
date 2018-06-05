function g = sigmoidGrad(z);
g = zeros(size(z));
g = sigmoid(z) .* (1-sigmoid(z));
end
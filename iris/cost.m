function [J grad] = cost(params,input_layer_size,hidden_layer_size,output_layer_size,X,y,lambda)

m = size(X,1);	

theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));
theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end),output_layer_size, (hidden_layer_size + 1));

I = eye(3);
J = 0;

theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));

for i=1:m,
	% Forward-Prop
	class = y(i,1);
	a1 = transpose(X(i,:));
	a1 = [1;a1];
	z2 = theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1;a2];
	z3 = theta2 * a2;
	a3 = sigmoid(z3);
	h = a3;
	J = J + (transpose(I(:,class)) * log(h)) + (transpose(1-I(:,class)) * (log(1-h)));
	% Back-Prop
	delta3 = a3 - I(:,class);
	delta2 = (transpose(theta2) * delta3) .* (a2 .* (1-a2));
	delta2 = delta2(2:end); % Missed Step
	theta2_grad = theta2_grad + (delta3 * transpose(a2));
	theta1_grad = theta1_grad + (delta2 * transpose(a1));
end;

% Regularization not Done

J = J * (-1/m);

theta1_grad = theta1_grad / m;
theta2_grad = theta2_grad / m;

grad = [theta1_grad(:) ; theta2_grad(:)];

end
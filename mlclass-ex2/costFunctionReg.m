function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta); % X have n+1 cols , h m * 1
j1 = -y' * log(h); % j1 1*1
j2 = (1-y)' * log(1 - h);

[len, dum] = size(theta);
J = 1/m * (j1 - j2) + lambda / (2*m) * (sumsq(theta(2:len)));

grad(1) = 1/m * ((h - y)' * X(:,1));

grad(2:len) = 1/m * ((h-y)' * X(:,2:len)) + lambda / m * (theta(2:len)');




% =============================================================

end

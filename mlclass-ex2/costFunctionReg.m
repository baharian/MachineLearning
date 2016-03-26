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

temp = X * theta;

mask = ones(size(theta)); mask(1) = 0;
sz = size(theta, 1);

%J = - sum(y .* log(sigmoid(temp)) + (1 - y) .* log(1 - sigmoid(temp))) / m + (lambda / (2 * m)) * (theta(2 : sz)' * theta(2 : sz));
J = - sum(y .* log(sigmoid(temp)) + (1 - y) .* log(1 - sigmoid(temp))) / m + (lambda / (2 * m)) * ((mask .* theta)' * (mask .* theta));
grad = ((sigmoid(temp) - y)' * X)' / m + (lambda / m) * (mask .* theta);

% =============================================================

end

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

h = sigmoid(X*theta);
a = -y'*log(h);         % when y = 1
b = (1-y')*log(1-h);    % when y = 0
theta(1) = 0;           % theta(1) not reg
reg_term = (lambda/(2*m)) * sum(theta.^2);   
J = (1/m) * sum(a-b) + reg_term;    % log.reg. costfunc

grad = (1/m) * X' * (h-y); % [3x100]*[100x1]
theta(1) = 0;               % theta(1) not reg
grad = grad + (lambda/m)*theta;

% =============================================================

end

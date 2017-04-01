function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Alternatively, going by classes 
% My technique (although more cumbersome, with yc)
%X = [ones(m,1) X];
%J = 0;
%a2 = [ ones(m,1) sigmoid(X*Theta1') ];
%for c = 1:num_labels
%  h = sigmoid( a2 * Theta2(c,:)' );
%  yc = y_matrix(:,c);
%  J = J + (1/m) * sum(-yc.*log(h) - (1-yc).*log(1-h));
%end

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);     % sets of [1x10] matrices

a1 = [ones(m,1) X];  % bias units +1
z2 = a1*Theta1'
a2 = sigmoid(z2);   % [5000x401]*[25x401]' = [5000x25]
a2 = [ones(m,1) a2]; % bias units +1
a3 = sigmoid(a2*Theta2');   % [5000x26] * [10x26]' = [5000x10]
h = a3;

% [5000x10].*[5000x10] -> sum -> [1x1]
J = (1/m) * sum(sum(-y_matrix.*log(h) - (1-y_matrix).*log(1-h)));   

% Regularization - Cost Function
% first columns of Theta1, Theta2 are bias terms - not regularized
Theta1(:,1) = zeros( size(Theta1,1), 1);
Theta2(:,1) = zeros( size(Theta2,1), 1);
J = J + (lambda/(2*m)) * (sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));


% GRADIENTS

d3 = ones(m, size(a3,2));
d2 = ones(m, size(a2,2)-1); % -bias

% for loop version (backprop only)
% Forum mentor: "The iterative method is very difficult to get working, and
% even if it does work, it runs about 50x slower than the vectorized
% method."
% (because re-forwardprop and backprop -- re-accessing...slow)

%for t = 1:m
    
    % [1x10]
%    d3(t,:) = a3(t,:) - y_matrix(t,:);
    % [1x10]*[10x25] .* [1x25]
%    d2(t,:) = d3(t,:) * Theta2(:,2:end) .* sigmoidGradient(z2(t,:));

%end


% VECTORIZATION
d3 = a3 - y_matrix;     % [5000x10]
% [5000x10]*[10x25] .* [5000x25]
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);
% accumulates
Delta1 = d2' * a1;  % [5000x25]'*[5000x401] = [25x401]
Delta2 = d3' * a2;   % [5000x10]'*[5000x26] = [10x26]

% scaling to gradient matrices
Theta1_grad = Delta1 * (1/m);
Theta2_grad = Delta2 * (1/m);

% Regularization
%Theta1(:,1) = zeros( size(Theta1,1), 1);
%Theta2(:,1) = zeros( size(Theta2,1), 1);
Theta1_grad = Theta1_grad + (lambda/m)*Theta1;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%C_minError = 0;
%sigma_minError = 0;
%error_previous = 0;

counter = 1;
C = 0.01;
for index_C = 1:10
    sigma = 0.01;
    for index_sigma = 1:10
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        
        if (counter ~= 1)
            if (error < error_previous)
                C_minError = C;
                sigma_minError = sigma;
                error_previous = error;
            end
        else
            C_minError = C;
            sigma_minError = sigma;
            error_previous = error;
        end

        sigma = sigma * 3;

        counter = counter + 1;
    end

    C = C * 3;
end

fprintf('trials: %f, error: %f', counter, error);

C = C_minError;
sigma = sigma_minError;

% =========================================================================

end

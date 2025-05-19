%% Load FFT Data
data = readtable('FFT_Normalized_Table.csv');

% Use first 32 FFT components (you must regenerate FFT with 32 points if your CSV has only 8)
fft_values = data{:, 1:8}; % change to 1:32 if available
theta = data.Theta;
phi = data.Phi;

% Inputs and targets
inputs = fft_values;
targets = [theta, phi];

% Standardize inputs
inputs = (inputs - mean(inputs)) ./ std(inputs);

% Split data
rng(42);
cv = cvpartition(size(inputs,1), 'HoldOut', 0.2);
XTrain = inputs(training(cv), :);
YTrain = targets(training(cv), :);
XTest = inputs(test(cv), :);
YTest = targets(test(cv), :);

%% ---- Deep Neural Network (DNN) with Dropout ----
layers = [
    featureInputLayer(size(XTrain, 2))
    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(2) % Theta and Phi
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', 32, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch');

net = trainNetwork(XTrain, YTrain, layers, options);

% Predict
YPredDNN = predict(net, XTest);

% Metrics
mse_dnn = mean((YPredDNN - YTest).^2, 'all');
r2_dnn = 1 - sum((YTest - YPredDNN).^2, 'all') / sum((YTest - mean(YTest)).^2, 'all');

fprintf('\n--- Deep Neural Network ---\n');
fprintf('MSE: %.4f\n', mse_dnn);
fprintf('R²: %.4f\n', r2_dnn);

% Save DNN results
dnn_results = array2table([YTest, YPredDNN], ...
    'VariableNames', {'Theta_Actual', 'Phi_Actual', 'Theta_Pred', 'Phi_Pred'});
writetable(dnn_results, 'Improved_DNN_Results.csv');

%% ---- Support Vector Regression (SVR) ----
svm_theta = fitrsvm(XTrain, YTrain(:,1), 'KernelFunction', 'gaussian');
svm_phi = fitrsvm(XTrain, YTrain(:,2), 'KernelFunction', 'gaussian');

theta_pred_svm = predict(svm_theta, XTest);
phi_pred_svm = predict(svm_phi, XTest);

svm_preds = [theta_pred_svm, phi_pred_svm];

% Metrics
mse_svm = mean((svm_preds - YTest).^2, 'all');
r2_svm = 1 - sum((YTest - svm_preds).^2, 'all') / sum((YTest - mean(YTest)).^2, 'all');

fprintf('\n--- Support Vector Regression (SVR) ---\n');
fprintf('MSE: %.4f\n', mse_svm);
fprintf('R²: %.4f\n', r2_svm);

% Save SVR results
svr_results = array2table([YTest, svm_preds], ...
    'VariableNames', {'Theta_Actual', 'Phi_Actual', 'Theta_Pred', 'Phi_Pred'});
writetable(svr_results, 'Improved_SVR_Results.csv');

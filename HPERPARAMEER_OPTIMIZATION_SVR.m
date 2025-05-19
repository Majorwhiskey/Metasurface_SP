clear; clc; close all;

% --- Load or Prepare your FFT feature data and normalized Theta, Phi targets ---
% For example, assume you already have:
%   inputs: Nx8 matrix of FFT features
%   outputs: Nx2 matrix of normalized [Theta, Phi]

% Replace below with your actual data loading/preparation
load('trained_RNN.mat');  % Assuming variables 'inputs' and 'outputs' are loaded
% inputs -> Nx8, outputs -> Nx2 (Theta, Phi normalized to [0,1])

% Verify sizes
disp(size(inputs));  % Should be [N_samples, 8]
disp(size(outputs)); % Should be [N_samples, 2]

% --- Use cvpartition for train/test split ---
cv = cvpartition(size(inputs,1), 'HoldOut', 0.2);

trainIdx = training(cv);
testIdx = test(cv);

XTrain = inputs(trainIdx, :);
YTrain = outputs(trainIdx, :);
XTest = inputs(testIdx, :);
YTest = outputs(testIdx, :);

% --- Train SVR models separately for Theta and Phi ---
% Theta model
svrTheta = fitrsvm(XTrain, YTrain(:,1), 'KernelFunction', 'rbf', 'Standardize', true);

% Phi model
svrPhi = fitrsvm(XTrain, YTrain(:,2), 'KernelFunction', 'rbf', 'Standardize', true);

% --- Predict on test data ---
thetaPred = predict(svrTheta, XTest);
phiPred = predict(svrPhi, XTest);

% --- Calculate Performance Metrics ---
mseTheta = mean((thetaPred - YTest(:,1)).^2);
msePhi = mean((phiPred - YTest(:,2)).^2);

% R-squared function
rsq = @(y_true,y_pred) 1 - sum((y_true - y_pred).^2)/sum((y_true - mean(y_true)).^2);

r2Theta = rsq(YTest(:,1), thetaPred);
r2Phi = rsq(YTest(:,2), phiPred);

% Display results
fprintf('--- SVR Model Performance ---\n');
fprintf('MSE Theta: %.4f | R^2 Theta: %.4f\n', mseTheta, r2Theta);
fprintf('MSE Phi: %.4f | R^2 Phi: %.4f\n', msePhi, r2Phi);

% --- Save results ---
results_table = table(YTest(:,1), YTest(:,2), thetaPred, phiPred, ...
    'VariableNames', {'Theta_Actual', 'Phi_Actual', 'Theta_Predicted', 'Phi_Predicted'});

writetable(results_table, 'SVR_Results.csv');
disp('Results saved to "SVR_Results.csv".');

% --- Optional: plot actual vs predicted ---
figure;
subplot(2,1,1);
plot(YTest(:,1), 'b-', 'DisplayName', 'Actual Theta'); hold on;
plot(thetaPred, 'r--', 'DisplayName', 'Predicted Theta');
legend('Location','best'); title('Theta: Actual vs Predicted'); grid on;

subplot(2,1,2);
plot(YTest(:,2), 'b-', 'DisplayName', 'Actual Phi'); hold on;
plot(phiPred, 'r--', 'DisplayName', 'Predicted Phi');
legend('Location','best'); title('Phi: Actual vs Predicted'); grid on;

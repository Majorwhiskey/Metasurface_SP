%% 1. Load and Preprocess Signal Data
clear; clc; close all;

% Load the received signal data
% Each row: [SNR, Theta, Phi, signal_samples...]
data = readmatrix('FFT_Normalized_Table.csv'); % Update path if needed

% Split metadata and signal
meta = data(:, 1:3); % SNR, Theta, Phi
signals = data(:, 4:end);

% Denoise each signal using a wavelet method
denoised_signals = zeros(size(signals));
for i = 1:size(signals, 1)
    denoised_signals(i, :) = wdenoise(signals(i, :), 2);
end

%% 2. Extract FFT Features (High-Resolution)
N_fft = 128;
fft_features = abs(fft(denoised_signals, N_fft, 2));
fft_features = fft_features(:, 1:N_fft/2); % Keep 64 features

% Normalize FFT Features
fft_features = normalize(fft_features);

% Normalize outputs
theta = meta(:, 2);
phi = meta(:, 3);
theta_norm = normalize(theta);
phi_norm = normalize(phi);
targets = [theta_norm, phi_norm];

%% 3. Split Dataset
rng(42); % For reproducibility
cv = cvpartition(size(fft_features,1), 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

XTrain = fft_features(idxTrain, :);
YTrain = targets(idxTrain, :);
XTest = fft_features(idxTest, :);
YTest = targets(idxTest, :);

%% 4. Train Gradient Boosted Trees for Regression
fprintf('--- Gradient Boosted Trees ---\n');
model_theta = fitrensemble(XTrain, YTrain(:,1), 'Method', 'LSBoost', 'NumLearningCycles', 200);
model_phi = fitrensemble(XTrain, YTrain(:,2), 'Method', 'LSBoost', 'NumLearningCycles', 200);

% Predict
pred_theta = predict(model_theta, XTest);
pred_phi = predict(model_phi, XTest);

% Metrics
mse_theta = mean((pred_theta - YTest(:,1)).^2);
mse_phi = mean((pred_phi - YTest(:,2)).^2);
r2_theta = 1 - sum((pred_theta - YTest(:,1)).^2)/sum((YTest(:,1) - mean(YTest(:,1))).^2);
r2_phi = 1 - sum((pred_phi - YTest(:,2)).^2)/sum((YTest(:,2) - mean(YTest(:,2))).^2);

fprintf('MSE Theta: %.4f | R²: %.4f\n', mse_theta, r2_theta);
fprintf('MSE Phi: %.4f | R²: %.4f\n', mse_phi, r2_phi);

%% 5. Save Results
results = table(YTest(:,1), YTest(:,2), pred_theta, pred_phi, ...
    'VariableNames', {'Theta_Actual', 'Phi_Actual', 'Theta_Predicted', 'Phi_Predicted'});
writetable(results, 'Improved_Ensemble_Results.csv');
fprintf('Results saved to \"Improved_Ensemble_Results.csv\".\n');

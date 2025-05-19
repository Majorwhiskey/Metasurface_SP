clear; clc; close all;

%% Parameters
Nx = 4; Ny = 4; d = 0.5; lambda = 1; k = 2*pi/lambda;
fs = 1e3; t = 0:1/fs:1; A = 1;
[x, y] = meshgrid(0:Nx-1, 0:Ny-1);
x = x * d; y = y * d;
signal = A * cos(2 * pi * 100 * t);

%% Generate received signals with noise for various SNR, theta, phi
snrVals = -5:2:5;
thetaVals = -90:5:90;
phiVals = -90:5:90;

received_signal_data = [];

for SNR = snrVals
    for theta = thetaVals
        for phi = phiVals
            AF = arrayFactor(theta, phi, Nx, Ny, x, y, k);
            rec_signal = AF * signal;
            noise = (1 / 10^(SNR/10)) * randn(size(rec_signal));
            noisy_signal = rec_signal + noise;
            received_signal_data = [received_signal_data; SNR, theta, phi, noisy_signal];
        end
    end
end

%% Feature extraction - 64-point FFT (magnitude)
numSignals = size(received_signal_data, 1);
Nfft = 64; % FFT length
fft_features = zeros(numSignals, Nfft);

for i = 1:numSignals
    sig = received_signal_data(i, 4:end);
    fft_val = fft(sig, Nfft);
    mag_fft = abs(fft_val);
    fft_features(i, :) = mag_fft;
end

%% Normalize features (per sample)
fft_features = (fft_features - min(fft_features,[],2)) ./ (max(fft_features,[],2) - min(fft_features,[],2) + eps);

%% Targets (Theta and Phi normalized)
theta_phi = received_signal_data(:, 2:3);
theta_norm = (theta_phi(:,1) - min(theta_phi(:,1))) / (max(theta_phi(:,1)) - min(theta_phi(:,1)));
phi_norm = (theta_phi(:,2) - min(theta_phi(:,2))) / (max(theta_phi(:,2)) - min(theta_phi(:,2)));
targets_norm = [theta_norm, phi_norm];

%% PCA for dimensionality reduction
[coeff, score, ~, ~, explained] = pca(fft_features);
cumExplained = cumsum(explained);
numComponents = find(cumExplained >= 95, 1);
reduced_features = score(:,1:numComponents);
fprintf('PCA reduced from %d to %d components (%.2f%% variance)\n', Nfft, numComponents, cumExplained(numComponents));

%% Split data 80% train / 20% test
rng(42);
numTrain = floor(0.8 * numSignals);
indices = randperm(numSignals);
trainIdx = indices(1:numTrain);
testIdx = indices(numTrain+1:end);

Xtrain = reduced_features(trainIdx, :);
Ytrain = targets_norm(trainIdx, :);
Xtest = reduced_features(testIdx, :);
Ytest = targets_norm(testIdx, :);

%% -------- Gradient Boosted Trees --------
template = templateTree('MaxNumSplits',20);
mdlTheta_ens = fitrensemble(Xtrain, Ytrain(:,1), 'Method', 'LSBoost', 'NumLearningCycles', 150, 'Learners', template);
mdlPhi_ens = fitrensemble(Xtrain, Ytrain(:,2), 'Method', 'LSBoost', 'NumLearningCycles', 150, 'Learners', template);

Ypred_theta_ens = predict(mdlTheta_ens, Xtest);
Ypred_phi_ens = predict(mdlPhi_ens, Xtest);

mse_theta_ens = mean((Ypred_theta_ens - Ytest(:,1)).^2);
mse_phi_ens = mean((Ypred_phi_ens - Ytest(:,2)).^2);
r2_theta_ens = 1 - sum((Ypred_theta_ens - Ytest(:,1)).^2) / sum((Ytest(:,1) - mean(Ytest(:,1))).^2);
r2_phi_ens = 1 - sum((Ypred_phi_ens - Ytest(:,2)).^2) / sum((Ytest(:,2) - mean(Ytest(:,2))).^2);

%% -------- Support Vector Regression (SVR) --------
mdlTheta_svr = fitrsvm(Xtrain, Ytrain(:,1), 'KernelFunction', 'rbf', 'Standardize', true);
mdlPhi_svr = fitrsvm(Xtrain, Ytrain(:,2), 'KernelFunction', 'rbf', 'Standardize', true);

Ypred_theta_svr = predict(mdlTheta_svr, Xtest);
Ypred_phi_svr = predict(mdlPhi_svr, Xtest);

mse_theta_svr = mean((Ypred_theta_svr - Ytest(:,1)).^2);
mse_phi_svr = mean((Ypred_phi_svr - Ytest(:,2)).^2);
r2_theta_svr = 1 - sum((Ypred_theta_svr - Ytest(:,1)).^2) / sum((Ytest(:,1) - mean(Ytest(:,1))).^2);
r2_phi_svr = 1 - sum((Ypred_phi_svr - Ytest(:,2)).^2) / sum((Ytest(:,2) - mean(Ytest(:,2))).^2);

%% -------- Feedforward Neural Network --------
% Normalize inputs (features already normalized, just double check)
Xtrain_nn = Xtrain';
Xtest_nn = Xtest';
Ytrain_nn = Ytrain';
Ytest_nn = Ytest';

hiddenLayerSize = 20;
net = fitnet(hiddenLayerSize, 'trainlm'); % Levenberg-Marquardt

net.trainParam.showWindow = false; % Disable GUI training window
net.trainParam.epochs = 200;
net.trainParam.goal = 1e-3;
net.trainParam.max_fail = 10;

% Train network
net = train(net, Xtrain_nn, Ytrain_nn);

% Predict
Ypred_nn = net(Xtest_nn);

% Metrics
mse_theta_nn = mean((Ypred_nn(1,:)' - Ytest(:,1)).^2);
mse_phi_nn = mean((Ypred_nn(2,:)' - Ytest(:,2)).^2);
r2_theta_nn = 1 - sum((Ypred_nn(1,:)' - Ytest(:,1)).^2) / sum((Ytest(:,1) - mean(Ytest(:,1))).^2);
r2_phi_nn = 1 - sum((Ypred_nn(2,:)' - Ytest(:,2)).^2) / sum((Ytest(:,2) - mean(Ytest(:,2))).^2);

%% Display results
fprintf('\n--- Gradient Boosted Trees ---\n');
fprintf('MSE Theta: %.4f | R² Theta: %.4f\n', mse_theta_ens, r2_theta_ens);
fprintf('MSE Phi: %.4f | R² Phi: %.4f\n', mse_phi_ens, r2_phi_ens);

fprintf('\n--- Support Vector Regression (SVR) ---\n');
fprintf('MSE Theta: %.4f | R² Theta: %.4f\n', mse_theta_svr, r2_theta_svr);
fprintf('MSE Phi: %.4f | R² Phi: %.4f\n', mse_phi_svr, r2_phi_svr);

fprintf('\n--- Neural Network ---\n');
fprintf('MSE Theta: %.4f | R² Theta: %.4f\n', mse_theta_nn, r2_theta_nn);
fprintf('MSE Phi: %.4f | R² Phi: %.4f\n', mse_phi_nn, r2_phi_nn);

%% Save all results to CSV (denormalize predictions)
theta_pred_ens = Ypred_theta_ens * (max(theta_phi(:,1)) - min(theta_phi(:,1))) + min(theta_phi(:,1));
phi_pred_ens = Ypred_phi_ens * (max(theta_phi(:,2)) - min(theta_phi(:,2))) + min(theta_phi(:,2));

theta_pred_svr = Ypred_theta_svr * (max(theta_phi(:,1)) - min(theta_phi(:,1))) + min(theta_phi(:,1));
phi_pred_svr = Ypred_phi_svr * (max(theta_phi(:,2)) - min(theta_phi(:,2))) + min(theta_phi(:,2));

theta_pred_nn = Ypred_nn(1,:)' * (max(theta_phi(:,1)) - min(theta_phi(:,1))) + min(theta_phi(:,1));
phi_pred_nn = Ypred_nn(2,:)' * (max(theta_phi(:,2)) - min(theta_phi(:,2))) + min(theta_phi(:,2));

theta_true = Ytest(:,1) * (max(theta_phi(:,1)) - min(theta_phi(:,1))) + min(theta_phi(:,1));
phi_true = Ytest(:,2) * (max(theta_phi(:,2)) - min(theta_phi(:,2))) + min(theta_phi(:,2));

results_table = table(theta_true, phi_true, ...
    theta_pred_ens, phi_pred_ens, ...
    theta_pred_svr, phi_pred_svr, ...
    theta_pred_nn, phi_pred_nn, ...
    'VariableNames', {'Theta_True','Phi_True', ...
    'Theta_Ens_Pred','Phi_Ens_Pred', ...
    'Theta_SVR_Pred','Phi_SVR_Pred', ...
    'Theta_NN_Pred','Phi_NN_Pred'});

writetable(results_table, 'All_Models_Results.csv');
disp('All model results saved to "All_Models_Results.csv".');

%% Plotting example: Actual vs predicted Theta for all models
figure;
plot(theta_true, 'k-', 'DisplayName', 'Actual Theta'); hold on;
plot(theta_pred_ens, 'r--', 'DisplayName', 'GBT Predicted Theta');
plot(theta_pred_svr, 'b--', 'DisplayName', 'SVR Predicted Theta');
plot(theta_pred_nn, 'g--', 'DisplayName', 'NN Predicted Theta');
xlabel('Sample Index'); ylabel('Theta (degrees)');
title('Actual vs Predicted Theta');
legend; grid on;

%% --- Supporting function ---
function AF = arrayFactor(theta, phi, Nx, Ny, x, y, k)
    AF = 0;
    for i = 1:Nx
        for j = 1:Ny
            phase_shift = k * (x(i,j)*sind(theta)*cosd(phi) + y(i,j)*sind(theta)*sind(phi));
            AF = AF + exp(1j * phase_shift);
        end
    end
end

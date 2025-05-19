clear; clc; close all;

%% Load and preprocess data
fft_table = readtable('FFT_Normalized_Table.csv');
fft_data = table2array(fft_table);

% Inputs: FFT values (columns 1-8)
X = fft_data(:, 1:8);
% Outputs: Theta and Phi (columns 9 and 10)
Y = fft_data(:, 9:10);

% Normalize inputs to [0,1]
X_min = min(X);
X_max = max(X);
X = (X - X_min) ./ (X_max - X_min + eps);

% Normalize outputs to [0,1]
Y_min = min(Y);
Y_max = max(Y);
Y = (Y - Y_min) ./ (Y_max - Y_min + eps);

% Shuffle data
N = size(X,1);
rng(42); % For reproducibility
idx = randperm(N);
X = X(idx,:);
Y = Y(idx,:);

% Split data: 80% training, 20% testing
split = round(0.8 * N);
X_train = X(1:split,:)';
Y_train = Y(1:split,:)';
X_test = X(split+1:end,:)';
Y_test = Y(split+1:end,:)';

%% Create and configure network
hiddenLayers = [32 16]; % Balanced network size
net = fitnet(hiddenLayers, 'trainlm'); % Use Levenberg-Marquardt for speed

% Training parameters
net.trainParam.epochs = 500;      % Moderate number of epochs
net.trainParam.goal = 1e-5;       % Training goal (MSE)
net.trainParam.showWindow = false; % Disable GUI window
net.trainParam.showCommandLine = true;

% Divide data manually (already split)
net.divideFcn = 'dividetrain'; % Use all given training data for training

%% Train network
[net, tr] = train(net, X_train, Y_train);

%% Predict on test data
Y_pred = net(X_test);

% Denormalize outputs back to original scale
Y_test_dn = Y_test' .* (Y_max - Y_min + eps) + Y_min;
Y_pred_dn = Y_pred' .* (Y_max - Y_min + eps) + Y_min;

%% Evaluate performance
mse_val = mean((Y_pred_dn - Y_test_dn).^2, 'all');
mae_val = mean(abs(Y_pred_dn - Y_test_dn), 'all');
r2_val = 1 - sum((Y_pred_dn - Y_test_dn).^2, 'all') / ...
             sum((Y_test_dn - mean(Y_test_dn, 'all')).^2, 'all');

fprintf('Performance Metrics:\n');
fprintf('MSE: %.6f\n', mse_val);
fprintf('MAE: %.6f\n', mae_val);
fprintf('R-squared (R^2): %.6f\n', r2_val);

%% Save network and results
save('balanced_FitNet.mat', 'net');

results_table = array2table([Y_test_dn, Y_pred_dn], ...
    'VariableNames', {'Theta_Actual', 'Phi_Actual', 'Theta_Predicted', 'Phi_Predicted'});
writetable(results_table, 'balanced_FitNet_Results.csv');
fprintf('Results saved as "balanced_FitNet_Results.csv".\n');

%% Visualization: Actual vs Predicted
figure;
subplot(2,1,1);
plot(Y_test_dn(:,1), 'b-', 'DisplayName', 'Actual Theta'); hold on;
plot(Y_pred_dn(:,1), 'r--', 'DisplayName', 'Predicted Theta');
xlabel('Sample Index'); ylabel('Theta');
title('Actual vs Predicted Theta');
legend('Location', 'best');
grid on; hold off;

subplot(2,1,2);
plot(Y_test_dn(:,2), 'b-', 'DisplayName', 'Actual Phi'); hold on;
plot(Y_pred_dn(:,2), 'r--', 'DisplayName', 'Predicted Phi');
xlabel('Sample Index'); ylabel('Phi');
title('Actual vs Predicted Phi');
legend('Location', 'best');
grid on; hold off;

%% Residuals plot
figure;
res_theta = Y_test_dn(:,1) - Y_pred_dn(:,1);
res_phi = Y_test_dn(:,2) - Y_pred_dn(:,2);

subplot(2,1,1);
plot(res_theta, 'g.', 'DisplayName', 'Residuals');
title('Residuals for Theta');
xlabel('Sample Index'); ylabel('Residual');
legend('Location', 'best');
grid on;

subplot(2,1,2);
plot(res_phi, 'g.', 'DisplayName', 'Residuals');
title('Residuals for Phi');
xlabel('Sample Index'); ylabel('Residual');
legend('Location', 'best');
grid on;

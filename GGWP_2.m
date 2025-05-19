clear; clc; close all;

%% Load the FFT data
fft_table = readtable('FFT_Normalized_Table.csv');

% Convert to array
fft_data = table2array(fft_table);

% Randomize rows
rng(42);
fft_data = fft_data(randperm(size(fft_data,1)), :);

% Split into input (X) and output (Y)
X = fft_data(:,1:8);           % Normalized FFT values
Y = fft_data(:,9:10);          % Theta and Phi

% Store original Y for later denormalization
Y_orig = Y;

% Normalize Y (Theta and Phi) to [0,1]
Y_min = min(Y);
Y_max = max(Y);
Y_norm = (Y - Y_min) ./ (Y_max - Y_min);

% Split into train/val/test
N = size(X,1);
train_end = floor(0.7*N);
val_end = floor(0.85*N);

X_train = X(1:train_end,:)';
Y_train = Y_norm(1:train_end,:)';

X_val = X(train_end+1:val_end,:)';
Y_val = Y_norm(train_end+1:val_end,:)';

X_test = X(val_end+1:end,:)';
Y_test = Y_norm(val_end+1:end,:)';

%% Create and Train Fitnet (Feedforward Neural Network)
hiddenSizes = [20 15]; % 2 hidden layers
net = fitnet(hiddenSizes, 'trainlm'); % Levenberg-Marquardt

% Set training parameters
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-5;
net.trainParam.showWindow = true;
net.divideFcn = 'divideind'; % We manually split data
net.divideParam.trainInd = 1:train_end;
net.divideParam.valInd = train_end+1:val_end;
net.divideParam.testInd = [];

% Train network
[net, tr] = train(net, X', Y_norm');

%% Predict on Test Data
Y_pred_norm = net(X_test);

% Denormalize predictions
Y_pred = Y_pred_norm' .* (Y_max - Y_min) + Y_min;
Y_actual = Y(val_end+1:end, :);

%% Evaluate Performance
mse_error = mean((Y_pred - Y_actual).^2, 'all');
mae_error = mean(abs(Y_pred - Y_actual), 'all');
r_squared = 1 - sum((Y_actual - Y_pred).^2, 'all') / sum((Y_actual - mean(Y_actual)).^2, 'all');

fprintf('\n--- Performance ---\n');
fprintf('MSE: %.6f\n', mse_error);
fprintf('MAE: %.6f\n', mae_error);
fprintf('R²: %.6f\n', r_squared);

%% Save results
results = array2table([Y_actual, Y_pred], ...
    'VariableNames', {'Theta_Actual', 'Phi_Actual', 'Theta_Predicted', 'Phi_Predicted'});
writetable(results, 'Improved_FitNet_Results.csv');
disp('Results saved to "Improved_FitNet_Results.csv"');

%% Visualization
figure;
subplot(2,1,1);
plot(Y_actual(:,1), 'b-', 'DisplayName', 'Actual \theta');
hold on;
plot(Y_pred(:,1), 'r--', 'DisplayName', 'Predicted \theta');
title('Actual vs Predicted \theta');
legend; grid on;

subplot(2,1,2);
plot(Y_actual(:,2), 'b-', 'DisplayName', 'Actual \phi');
hold on;
plot(Y_pred(:,2), 'r--', 'DisplayName', 'Predicted \phi');
title('Actual vs Predicted \phi');
legend; grid on;

% Residuals
figure;
subplot(2,1,1);
res_theta = Y_actual(:,1) - Y_pred(:,1);
plot(res_theta, 'g.');
title('Residuals for \theta');
ylabel('Residual'); grid on;

subplot(2,1,2);
res_phi = Y_actual(:,2) - Y_pred(:,2);
plot(res_phi, 'g.');
title('Residuals for \phi');
ylabel('Residual'); grid on;

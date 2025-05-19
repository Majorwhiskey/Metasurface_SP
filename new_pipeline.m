clear; clc; close all;

% Parameters (same as your original)
Nx = 4; Ny = 4; d = 0.5; lambda = 1; k = 2*pi/lambda;
fs = 1e3; t = 0:1/fs:1; A = 1;
[x, y] = meshgrid(0:Nx-1, 0:Ny-1);
x = x * d; y = y * d;
signal = A * cos(2*pi*100*t);

% Parameters for FFT and data collection
NFFT = 16;  % Increased FFT length
received_signal_data = [];

% For demo, smaller parameter grid (you can increase for more data)
SNR_vals = -5:5:5;
theta_vals = -30:15:30;
phi_vals = -30:15:30;

% Generate dataset
for SNR = SNR_vals
    for theta = theta_vals
        for phi = phi_vals
            received_signal = calculate_signal(SNR, theta, phi, Nx, Ny, x, y, signal, k, t);
            % Take first NFFT samples (adjust if signal length < NFFT)
            data_segment = received_signal(1:NFFT);
            received_signal_data = [received_signal_data; SNR, theta, phi, data_segment];
        end
    end
end

% Feature extraction: magnitude + phase of FFT
num_samples = size(received_signal_data,1);
features = zeros(num_samples, NFFT*2);
labels = zeros(num_samples, 2);

for i = 1:num_samples
    segment = received_signal_data(i, 4:4+NFFT-1);
    fft_vals = fft(segment, NFFT);
    mag = abs(fft_vals);
    ph = angle(fft_vals);
    features(i, :) = [mag, ph];
    labels(i, :) = received_signal_data(i, 2:3); % theta, phi
end

% Normalize features (0-1 scaling)
features = (features - min(features)) ./ (max(features) - min(features) + eps);

% Normalize labels to [0,1]
labels = (labels - min(labels)) ./ (max(labels) - min(labels) + eps);

% Shuffle data
rng(123); % reproducibility
idx = randperm(num_samples);
features = features(idx, :);
labels = labels(idx, :);

% Train-test split (80/20)
split_idx = floor(0.8 * num_samples);
trainX = features(1:split_idx, :)';
trainY = labels(1:split_idx, :)';
testX = features(split_idx+1:end, :)';
testY = labels(split_idx+1:end, :)';

% Define a deep fully connected network (FitNet)
hiddenLayerSizes = [50, 30]; % 2 layers with 50 and 30 neurons
net = fitnet(hiddenLayerSizes, 'trainlm');
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-5;
net.divideFcn = 'dividerand'; % further divide train data for validation
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;

% Train the network
[net, tr] = train(net, trainX, trainY);

% Predict on test data
predY = net(testX);

% Denormalize predictions and test labels
minLabels = min(received_signal_data(:,2:3));
maxLabels = max(received_signal_data(:,2:3));
predY_denorm = predY' .* (maxLabels - minLabels + eps) + minLabels;
testY_denorm = testY' .* (maxLabels - minLabels + eps) + minLabels;

% Calculate R^2 for each output
R2_Theta = 1 - sum((testY_denorm(:,1) - predY_denorm(:,1)).^2) / sum((testY_denorm(:,1) - mean(testY_denorm(:,1))).^2);
R2_Phi = 1 - sum((testY_denorm(:,2) - predY_denorm(:,2)).^2) / sum((testY_denorm(:,2) - mean(testY_denorm(:,2))).^2);

fprintf('Deep NN R^2 for Theta: %.4f\n', R2_Theta);
fprintf('Deep NN R^2 for Phi: %.4f\n', R2_Phi);

% Save results
results_table = array2table([testY_denorm, predY_denorm], ...
    'VariableNames', {'Theta_Actual', 'Phi_Actual', 'Theta_Predicted', 'Phi_Predicted'});
writetable(results_table, 'Improved_FitNet_Results.csv');
disp('Results saved to "Improved_FitNet_Results.csv".');

% Supporting function definitions (same as before) ------------------------

function signal_out = calculate_signal(SNR, theta, phi, Nx, Ny, x, y, signal, k, t)
    Array_factor = ArrayFactor(theta, phi, Nx, Ny, x, y, k);
    received_signal = Array_factor * signal;  
    noise = (1 / 10^(SNR / 10)) * randn(size(received_signal));  
    signal_out = received_signal + noise;  
end

function Array_factor = ArrayFactor(theta, phi, Nx, Ny, x, y, k)
    Array_factor = 0;
    for i = 1:Nx
        for j = 1:Ny
            phase_shift = k * (x(i,j) * sin(deg2rad(theta)) * cos(deg2rad(phi)) + ...
                               y(i,j) * sin(deg2rad(theta)) * sin(deg2rad(phi)));
            Array_factor = Array_factor + exp(1j * phase_shift);
        end
    end
end

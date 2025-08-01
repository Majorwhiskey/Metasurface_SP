%% 8-point Radix-2 FFT for all rows and plot on a single graph with specified legend entries
clear all;
clc;
close all;

% Rectangular antenna/element array for metasurface

% Parameters
Nx = 4; % Number of elements in the x-dimension
Ny = 4; % Number of elements in the y-dimension
d = 0.5; % Distance between elements in wavelengths
lambda = 1; % Wavelength
k = 2 * pi / lambda; % Wave number
fs = 1e3; % Sampling frequency
t = 0:1/fs:1; % Time vector
A = 1; % Amplitude of the incoming signal

% Create grid of element positions
[x, y] = meshgrid(0:Nx-1, 0:Ny-1);
x = x * d;
y = y * d;

% Generate the incoming signal (a simple sinusoidal wave in this example)
signal = A * cos(2 * pi * 100 * t);  % Signal as a cosine wave at 100 Hz

% Initialize the received signal matrix
received_signal_data = []; % Store received signal data

% Total number of iterations (for progress display)
total_iterations = length(-5:2:5) * length(-90:5:90) * length(-90:5:90);
current_iteration = 0;

% Loop over different SNR, theta, phi values
for SNR = -5:2:5  % Reduce the step size for theta and phi
    for theta = -90:5:90  % Increased step size
        for phi = -90:5:90  % Increased step size
            % Update progress display
            current_iteration = current_iteration + 1;
            disp(['Processing: ', num2str(current_iteration), ' of ', num2str(total_iterations)]);
            
            % Calculate the received signal for this combination of parameters
            received_signal = calculate_signal(SNR, theta, phi, Nx, Ny, x, y, signal, k, t);
            
            % Store the received signal data
            received_signal_data = [received_signal_data; SNR, theta, phi, received_signal];
        end
    end
end

% Function to calculate the signal based on SNR, theta, and phi
function signal_out = calculate_signal(SNR, theta, phi, Nx, Ny, x, y, signal, k, t)
    % Calculate Array Factor
    Array_factor = ArrayFactor(theta, phi, Nx, Ny, x, y, k);
    
    % Ensure Array_factor is a scalar for multiplication with the signal
    % Signal is a time-domain signal, so multiply it with the scalar Array_factor
    received_signal = Array_factor * signal;  % Multiply by signal
    
    % Adjust received signal by the SNR (simple model)
    noise = (1 / 10^(SNR / 10)) * randn(size(received_signal));  % Add noise based on SNR
    signal_out = received_signal + noise;  % Add noise to the signal
end

% Function to calculate the Array Factor
function Array_factor = ArrayFactor(theta, phi, Nx, Ny, x, y, k)
    % Initialize Array Factor
    Array_factor = 0;
    
    % Calculate the array factor based on the positions of the elements
    for i = 1:Nx
        for j = 1:Ny
            % Calculate the phase shift for element (i,j) due to its position
            phase_shift = k * (x(i,j) * sin(deg2rad(theta)) * cos(deg2rad(phi)) + ...
                               y(i,j) * sin(deg2rad(theta)) * sin(deg2rad(phi)));
            
            % Add the contribution from this element to the array factor
            Array_factor = Array_factor + exp(1j * phase_shift);
        end
    end
    
    % Normalize the Array Factor (optional, if needed)
    % Array_factor = Array_factor / (Nx * Ny); % Uncomment if you want to normalize
end

% Initialize table to store FFT values, theta, and phi
fft_table_data = [];

% Process each row of received signal data
for row = 1:size(received_signal_data, 1)
    % Extract FFT data for the current row (first 8 FFT values)
    fft_values = abs(fft(received_signal_data(row, 4:11), 8));
    
    % Normalize the FFT values to the range [0, 1]
    normalized_fft_values = (fft_values - min(fft_values)) / (max(fft_values) - min(fft_values));
    
    % Extract theta and phi values
    theta = received_signal_data(row, 2); % Theta value
    phi = received_signal_data(row, 3);   % Phi value
    
    % Combine the normalized FFT values with theta and phi
    fft_table_data = [fft_table_data; normalized_fft_values, theta, phi];
end

% Create a table for the data
fft_table = array2table(fft_table_data, ...
    'VariableNames', {'FFT_1', 'FFT_2', 'FFT_3', 'FFT_4', ...
                      'FFT_5', 'FFT_6', 'FFT_7', 'FFT_8', ...
                      'Theta', 'Phi'});

% Display the first 10 rows of the table for verification
disp(fft_table(1:10, :)); % Show the first 10 rows

% Optionally, export the table to a CSV file for further use
writetable(fft_table, 'FFT_Normalized_Table.csv');
disp('Table saved as "FFT_Normalized_Table.csv".');

%% Data Preparation
% Randomize rows in a non-uniform fashion
rng(42); % Set random seed for reproducibility
randomized_indices = randperm(size(fft_table_data, 1));
fft_table_data = fft_table_data(randomized_indices, :);

% Split data into inputs (FFT values) and outputs (Theta and Phi)
inputs = fft_table_data(:, 1:8); % FFT values
outputs = fft_table_data(:, 9:10); % Theta and Phi

% Normalize Theta and Phi to range [0, 1] for ANN training
outputs = (outputs - min(outputs, [], 1)) ./ (max(outputs, [], 1) - min(outputs, [], 1));

% Split data into training and testing sets (80% for training)
split_ratio = 0.8;
num_training_samples = floor(split_ratio * size(inputs, 1));

training_inputs = inputs(1:num_training_samples, :);
training_outputs = outputs(1:num_training_samples, :);

testing_inputs = inputs(num_training_samples+1:end, :);
testing_outputs = outputs(num_training_samples+1:end, :);
%% Create and Train the Recurrent Neural Network
% Define an RNN with 300 hidden neurons and feedback connections
layerDelays = 1:2; % Delays for the input layer
hiddenSizes = 10; % Number of hidden neurons
trainFcn = 'trainlm';

% Create a layer recurrent network
net = layrecnet(layerDelays,hiddenSizes,trainFcn);

% Configure network training options
net.trainFcn = 'trainlm'; % Use Levenberg-Marquardt algorithm
net.trainParam.epochs = 500; % Maximum number of epochs
net.trainParam.goal = 1e-5; % Training goal
net.trainParam.showCommandLine = true; % Display training progress in the command line
net.trainParam.showWindow = false; % Disable GUI window

% Prepare the data as sequences
training_inputs_seq = con2seq(training_inputs'); % Convert training inputs to sequences
training_outputs_seq = con2seq(training_outputs'); % Convert training outputs to sequences

% Train the RNN
[net, tr] = train(net, training_inputs_seq, training_outputs_seq);

%% Evaluate the Recurrent Neural Network
% Prepare testing data as sequences
testing_inputs_seq = con2seq(testing_inputs'); % Convert testing inputs to sequences

% Predict outputs using the trained RNN
predicted_outputs_seq = net(testing_inputs_seq);

% Convert predicted outputs back to matrix form
predicted_outputs = cell2mat(predicted_outputs_seq)';

% Calculate performance metrics
mse_error = mean((predicted_outputs - testing_outputs).^2, 'all');
mae_error = mean(abs(predicted_outputs - testing_outputs), 'all');

% Calculate R-squared (coefficient of determination)
ss_res = sum((testing_outputs - predicted_outputs).^2, 'all');
ss_tot = sum((testing_outputs - mean(testing_outputs, 'all')).^2, 'all');
r_squared = 1 - (ss_res / ss_tot);

% Display performance metrics
disp('Performance Metrics:');
disp(['Mean Squared Error (MSE): ', num2str(mse_error)]);
disp(['Mean Absolute Error (MAE): ', num2str(mae_error)]);
disp(['R-squared (R^2): ', num2str(r_squared)]);

%% Save the Network and Results
% Save the trained RNN for later use
save('trained_RNN.mat', 'net');

% Save the predictions and actual values for further analysis
results_table = array2table([testing_outputs, predicted_outputs], ...
    'VariableNames', {'Theta_Actual', 'Phi_Actual', 'Theta_Predicted', 'Phi_Predicted'});
writetable(results_table, 'RNN_Results.csv');
disp('Results saved as "RNN_Results.csv".');

%% Visualize Performance
% Plot actual vs predicted values for Theta
figure;
subplot(2, 1, 1);
plot(testing_outputs(:, 1), 'b-', 'DisplayName', 'Actual Theta'); % Actual in blue
hold on;
plot(predicted_outputs(:, 1), 'r--', 'DisplayName', 'Predicted Theta'); % Predicted in red
xlabel('Sample Index');
ylabel('Theta');
title('Actual vs Predicted Theta');
legend('Location', 'best');
grid on;
hold off;

% Plot actual vs predicted values for Phi
subplot(2, 1, 2);
plot(testing_outputs(:, 2), 'b-', 'DisplayName', 'Actual Phi'); % Actual in blue
hold on;
plot(predicted_outputs(:, 2), 'r--', 'DisplayName', 'Predicted Phi'); % Predicted in red
xlabel('Sample Index');
ylabel('Phi');
title('Actual vs Predicted Phi');
legend('Location', 'best');
grid on;
hold off;


% Display residuals
figure;
residuals_theta = testing_outputs(:, 1) - predicted_outputs(:, 1);
residuals_phi = testing_outputs(:, 2) - predicted_outputs(:, 2);

subplot(2, 1, 1);
plot(residuals_theta, 'g.', 'DisplayName', 'Residuals'); % Residuals in green
title('Residuals for Theta');
xlabel('Sample Index');
ylabel('Residual');
legend('Location', 'best');
grid on;

subplot(2, 1, 2);
plot(residuals_phi, 'g.', 'DisplayName', 'Residuals'); % Residuals in green
title('Residuals for Phi');
xlabel('Sample Index');
ylabel('Residual');
legend('Location', 'best');
grid on;


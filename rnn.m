%% Create and Train the Recurrent Neural Network
% Define an RNN with 300 hidden neurons and feedback connections
inputDelays = 1:2; % Delays for the input layer
feedbackDelays = 1:2; % Delays for the feedback layer
hiddenLayerSize = 300; % Number of hidden neurons

% Create a layer recurrent network
net = layrecnet(inputDelays, feedbackDelays, hiddenLayerSize);

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
plot(testing_outputs(:, 1), predicted_outputs(:, 1), 'o');
xlabel('Actual Theta');
ylabel('Predicted Theta');
title('Actual vs Predicted Theta');
grid on;

% Plot actual vs predicted values for Phi
subplot(2, 1, 2);
plot(testing_outputs(:, 2), predicted_outputs(:, 2), 'o');
xlabel('Actual Phi');
ylabel('Predicted Phi');
title('Actual vs Predicted Phi');
grid on;

% Display residuals
figure;
residuals_theta = testing_outputs(:, 1) - predicted_outputs(:, 1);
residuals_phi = testing_outputs(:, 2) - predicted_outputs(:, 2);

subplot(2, 1, 1);
plot(residuals_theta, 'o');
title('Residuals for Theta');
xlabel('Sample Index');
ylabel('Residual');
grid on;

subplot(2, 1, 2);
plot(residuals_phi, 'o');
title('Residuals for Phi');
xlabel('Sample Index');
ylabel('Residual');
grid on;

% Load data again
fft_table = readtable('FFT_Normalized_Table.csv');
data = table2array(fft_table);

X = data(:,1:8);
Y = data(:,9:10);

% Shuffle
rng(42);
idx = randperm(size(X,1));
X = X(idx,:);
Y = Y(idx,:);

% Split 80% train, 20% test
split = round(0.8 * size(X,1));
X_train = X(1:split,:);
Y_train = Y(1:split,:);
X_test = X(split+1:end,:);
Y_test = Y(split+1:end,:);

% Train ensemble regression models for Theta and Phi separately
model_theta = fitrensemble(X_train, Y_train(:,1), 'Method','Bag');
model_phi = fitrensemble(X_train, Y_train(:,2), 'Method','Bag');

% Predict
Y_pred_theta = predict(model_theta, X_test);
Y_pred_phi = predict(model_phi, X_test);

% Calculate R^2
R2_theta = 1 - sum((Y_test(:,1) - Y_pred_theta).^2)/sum((Y_test(:,1)-mean(Y_test(:,1))).^2);
R2_phi = 1 - sum((Y_test(:,2) - Y_pred_phi).^2)/sum((Y_test(:,2)-mean(Y_test(:,2))).^2);

fprintf('Ensemble Regression R^2 for Theta: %.4f\n', R2_theta);
fprintf('Ensemble Regression R^2 for Phi: %.4f\n', R2_phi);

% Save results
results = table(Y_test(:,1), Y_test(:,2), Y_pred_theta, Y_pred_phi, ...
    'VariableNames', {'Theta_Actual','Phi_Actual','Theta_Predicted','Phi_Predicted'});
writetable(results,'Ensemble_Results.csv');
disp('Ensemble model results saved as Ensemble_Results.csv');

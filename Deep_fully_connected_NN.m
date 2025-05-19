% Load and prepare data
fft_table = readtable('FFT_Normalized_Table.csv');
data = table2array(fft_table);

X = data(:,1:8)';
Y = data(:,9:10)';

% Normalize inputs and outputs (min-max)
X_min = min(X,[],2);
X_max = max(X,[],2);
X = (X - X_min) ./ (X_max - X_min + eps);

Y_min = min(Y,[],2);
Y_max = max(Y,[],2);
Y = (Y - Y_min) ./ (Y_max - Y_min + eps);

% Split indices
N = size(X,2);
rng(42);
idx = randperm(N);
split = round(0.8 * N);

XTrain = X(:,idx(1:split));
YTrain = Y(:,idx(1:split));
XTest = X(:,idx(split+1:end));
YTest = Y(:,idx(split+1:end));

% Define layers
layers = [
    featureInputLayer(8,"Normalization","none")
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(2)
    regressionLayer];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'MiniBatchSize', 64, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

% Train the network
net = trainNetwork(XTrain', YTrain', layers, options);

% Predict on test data
YPred = predict(net, XTest');

% Denormalize
YTest_dn = YTest .* (Y_max - Y_min + eps) + Y_min;
YPred_dn = YPred' .* (Y_max - Y_min + eps) + Y_min;

% Calculate R^2
R2_theta = 1 - sum((YTest_dn(:,1) - YPred_dn(:,1)).^2)/sum((YTest_dn(:,1)-mean(YTest_dn(:,1))).^2);
R2_phi = 1 - sum((YTest_dn(:,2) - YPred_dn(:,2)).^2)/sum((YTest_dn(:,2)-mean(YTest_dn(:,2))).^2);

fprintf('Deep NN R^2 for Theta: %.4f\n', R2_theta);
fprintf('Deep NN R^2 for Phi: %.4f\n', R2_phi);

% Save results
results = table(YTest_dn(:,1), YTest_dn(:,2), YPred_dn(:,1), YPred_dn(:,2), ...
    'VariableNames', {'Theta_Actual','Phi_Actual','Theta_Predicted','Phi_Predicted'});
writetable(results,'DeepNN_Results.csv');
disp('Deep NN results saved as DeepNN_Results.csv');

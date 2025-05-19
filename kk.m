% Define parameters
layerDelays = 1:2;
hiddenSizes = [10 10];
trainFcn = 'trainlm'; % Training function as a string

% Create a layer recurrent network
net = layrecnet(layerDelays, hiddenSizes, trainFcn);

% Display network structure
view(net);

% Example data for training
[X, T] = simpleseries_dataset;

% Prepare data
[Xs, Xi, Ai, Ts] = preparets(net, X, T);

% Train the network
net = train(net, Xs, Ts, Xi, Ai);

clear
close all

%% Observer and Action
% Open Simulink
mdl = "rlwatertank";
open_system(mdl);

% Action property
actionInfo = rlNumericSpec([1 1], ...
    'LowerLimit', -10, ...
    'UpperLimit',  10);
actionInfo.Name = "flow";

% Observation property
observationInfo = rlNumericSpec([3 1], ...
    'LowerLimit', [-inf -inf 0]', ...
    'UpperLimit', [inf inf inf]');
observationInfo.Name = "observations";
observationInfo.Description = "integrated error, error, and measured height";

%% create the env for RL
env = rlSimulinkEnv(mdl, mdl + "/RL Agent", observationInfo, actionInfo);
env.ResetFcn = @(in)localResetFcn(in);

function in = localResetFcn(in)
    h = 3*randn + 10;
    while h <= 0 || h >= 20
        h = 3*randn + 10;
    end
    in = setBlockParameter(in, ...
        "rlwatertank/Desired Water Level", ...
        Value=num2str(h));

    h = 3*randn + 10;
    while h <= 0 || h >= 20
        h = 3*randn + 10;
    end
    in = setBlockParameter(in, ...
        "rlwatertank/Water-Tank System/H", ...
        InitialCondition=num2str(h));
end

% Define Actor Network
actorNet = [
    featureInputLayer(observationInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(24, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(24, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numel(actionInfo.LowerLimit), 'Name', 'fc3')
    tanhLayer('Name', 'tanh')
    scalingLayer('Name', 'scale', 'Scale', 10)
];
actorNetwork = dlnetwork(actorNet);
actor = rlContinuousDeterministicActor(actorNetwork, observationInfo, actionInfo);

% Define Critic Network
criticStatePath = [
    featureInputLayer(observationInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(24, 'Name', 'criticStateFC1')
    reluLayer('Name', 'criticRelu1')
];
criticActionPath = [
    featureInputLayer(actionInfo.Dimension(1), 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(24, 'Name', 'criticActionFC1')
];
criticCommonPath = [
    additionLayer(2, 'Name', 'add')
    fullyConnectedLayer(24, 'Name', 'criticCommonFC1')
    reluLayer('Name', 'criticRelu2')
    fullyConnectedLayer(1, 'Name', 'criticOutput') % calculate Q value
];
criticNet = layerGraph(criticStatePath);
criticNet = addLayers(criticNet, criticActionPath);
criticNet = addLayers(criticNet, criticCommonPath);
criticNet = connectLayers(criticNet, 'criticRelu1', 'add/in1');
criticNet = connectLayers(criticNet, 'criticActionFC1', 'add/in2'); 
criticNetwork = dlnetwork(criticNet);
critic = rlQValueFunction(criticNetwork, observationInfo, actionInfo);

% Set DDPG agent
agentOptions = rlDDPGAgentOptions(...
    'SampleTime', 0.1, ...
    'DiscountFactor', 0.99); % the more closer to 1, more concentrate on the future
agent = rlDDPGAgent(actor, critic, agentOptions);

% Training
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 500, ...
    'MaxStepsPerEpisode', 2000, ...
    'ScoreAveragingWindowLength', 10, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeReward', ...
    'StopTrainingValue', 500);

trainingStats = train(agent, env, trainOpts);

%% Test the result
env.ResetFcn = [];

simOptions = rlSimulationOptions('MaxSteps', 2000);
experience = sim(env, agent, simOptions);

% in case of further training
env.ResetFcn = @(in)localResetFcn(in);
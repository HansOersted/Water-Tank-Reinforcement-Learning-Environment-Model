% 打开 Simulink 模型
mdl = "rlwatertank";
open_system(mdl);

% 动作规格
actionInfo = rlNumericSpec([1 1], ...
    'LowerLimit', 0, ...
    'UpperLimit', 1);
actionInfo.Name = "flow";

% 观测规格
observationInfo = rlNumericSpec([3 1], ...
    'LowerLimit', [-inf -inf 0]', ...
    'UpperLimit', [inf inf inf]');
observationInfo.Name = "observations";
observationInfo.Description = "integrated error, error, and measured height";

% 创建强化学习环境
env = rlSimulinkEnv(mdl, mdl + "/RL Agent", observationInfo, actionInfo);

% 设置重置函数
env.ResetFcn = @(in)localResetFcn(in);

% 定义重置函数
function in = localResetFcn(in)
    % 随机化参考信号
    h = 3*randn + 10;
    while h <= 0 || h >= 20
        h = 3*randn + 10;
    end
    in = setBlockParameter(in, ...
        "rlwatertank/Desired Water Level", ...
        Value=num2str(h));

    % 随机化初始液位
    h = 3*randn + 10;
    while h <= 0 || h >= 20
        h = 3*randn + 10;
    end
    in = setBlockParameter(in, ...
        "rlwatertank/Water-Tank System/H", ...
        InitialCondition=num2str(h));
end

% 定义 Actor 网络
actorNet = [
    featureInputLayer(observationInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(24, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(24, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(numel(actionInfo.LowerLimit), 'Name', 'fc3')
    tanhLayer('Name', 'tanh')
    scalingLayer('Name', 'scale', 'Scale', 1)
];
actorNetwork = dlnetwork(actorNet);
actor = rlContinuousDeterministicActor(actorNetwork, observationInfo, actionInfo);

% 定义修正后的 Critic 网络
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
    fullyConnectedLayer(1, 'Name', 'criticOutput') % 输出单一 Q 值
];
criticNet = layerGraph(criticStatePath);
criticNet = addLayers(criticNet, criticActionPath);
criticNet = addLayers(criticNet, criticCommonPath);
criticNet = connectLayers(criticNet, 'criticRelu1', 'add/in1'); % 修正连接
criticNet = connectLayers(criticNet, 'criticActionFC1', 'add/in2'); % 修正连接
criticNetwork = dlnetwork(criticNet);
critic = rlQValueFunction(criticNetwork, observationInfo, actionInfo);

% 配置 DDPG 智能体选项
agentOptions = rlDDPGAgentOptions(...
    'SampleTime', 0.1, ...
    'DiscountFactor', 0.99);
agent = rlDDPGAgent(actor, critic, agentOptions);

% 配置训练选项
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 500, ...
    'MaxStepsPerEpisode', 200, ...
    'ScoreAveragingWindowLength', 10, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeReward', ...
    'StopTrainingValue', 500);

% 开始训练
trainingStats = train(agent, env, trainOpts);

% 测试智能体
simOptions = rlSimulationOptions('MaxSteps', 2000);
experience = sim(env, agent, simOptions);

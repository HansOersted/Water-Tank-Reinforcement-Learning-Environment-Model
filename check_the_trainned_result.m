% Check the Trained Result Script
clear; clc; close all;

% Load Simulink Model
mdl = "rlwatertank";
open_system(mdl);

% Load Trained Agent
load('trainedAgent.mat', 'agent');

% Action Specifications
actionInfo = rlNumericSpec([1 1], 'LowerLimit', 0, 'UpperLimit', 1);
actionInfo.Name = "flow";

% Observation Specifications
observationInfo = rlNumericSpec([3 1], ...
    'LowerLimit', [-inf; -inf; 0], ...
    'UpperLimit', [inf; inf; 20]);
observationInfo.Name = "observations";

% Create Environment
env = rlSimulinkEnv(mdl, mdl + "/RL Agent", observationInfo, actionInfo);
env.ResetFcn = @(in)localResetFcn(in);

% Simulation Options
simOptions = rlSimulationOptions('MaxSteps', 400);

% Simulate Trained Agent
experience = sim(env, agent, simOptions);

% Local Reset Function
function in = localResetFcn(in)
    h = 10 + randn * 3;
    while h <= 0 || h >= 20
        h = 10 + randn * 3;
    end
    in = setBlockParameter(in, "rlwatertank/Desired Water Level", "Value", num2str(h));
    h = 10 + randn * 3;
    while h <= 0 || h >= 20
        h = 10 + randn * 3;
    end
    in = setBlockParameter(in, "rlwatertank/Water-Tank System/H", "InitialCondition", num2str(h));
end

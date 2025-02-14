# Water Tank Reinforcement Learning Control

This repo is about an open demo from MathWorks `openExample('rl/WaterTankEnvironmentModelExample')`.  
The agent is not well-defined by Mathworks and will be decided in this repo.

## 1. Simulink settings in agent

![image](https://github.com/user-attachments/assets/670591df-5af6-4dc2-9923-7f9360a83dd3)

The observation, reward, and isdone are specified in Simulink.  
This [document](https://se.mathworks.com/help/reinforcement-learning/ug/water-tank-simulink-reinforcement-learning-environment.html), from MathWorks,
 details the specifications and the desinations of each block in the Simulink.

Interestingly, the system dynamics are not explicit for the controller; the RL-based controller is not relying on the model.

## 2. Training function

The final function used to train the model in m file is 

```
trainingStats = train(agent, env, trainOpts);
```

where `trainOpts` can be set by `rlTrainingOptions()`.

The rest two arguments, `env` and `agent`, are specified in the next two sections.

## 3. Environment setting (`env`)

`env` specifies the observation information and the action information.  
These two informations include the upper and lower bounds, dimensions, etc.

## 4. Agent setting (`agent`)

The agent in this repo uses DDPG policy. It is initialized with

```
agent = rlDDPGAgent(actor, critic, agentOptions);
```

`actor` and `critic` are equipped based on Neural Networks to take actions and make evaluations, respectively.

```
actor = rlContinuousDeterministicActor(actorNetwork, observationInfo, actionInfo);
```

```
critic = rlQValueFunction(criticNetwork, observationInfo, actionInfo);
```

Note that `Q` value is calculated based on the current state and action.   
And it is calculated by combining action and state.

```
criticNet = connectLayers(criticNet, 'criticRelu1', 'add/in1');
criticNet = connectLayers(criticNet, 'criticActionFC1', 'add/in2');
```

## 5. Simulation result

The tracking result is visualized in the figure below.  
The response contains the oscillations and overshoot, which can be avoided by the further tune on the Neural Network.

![image](https://github.com/user-attachments/assets/27327c28-a712-4297-a672-c125a8da7663)

The training history can be traced below.

![image](https://github.com/user-attachments/assets/1385b67c-91ff-4226-8324-97ccc5128c73)

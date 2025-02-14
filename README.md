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




![image](https://github.com/user-attachments/assets/27327c28-a712-4297-a672-c125a8da7663)
![image](https://github.com/user-attachments/assets/1385b67c-91ff-4226-8324-97ccc5128c73)

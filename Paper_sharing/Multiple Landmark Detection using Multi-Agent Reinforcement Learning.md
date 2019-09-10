
Multiple Landmark Detection using Multi-Agent Reinforcement Learning
===
[TOC]

## Introduction
>Automatic detection of ***anatomical landmarks*** is an important step for a wide range of applications in medical image analysis. The location of anatomical landmarks is interdependent and non-random in a human anatomy, hence locating one is able to help locate others. In this project, the author formulate the landmark detection problem as a cocurrent ***partially observable markov decision process*** (POMDP) navigating in a medical image environment towards the target landmarks. The author create a ***collaborative Deep Q-Network*** (DQN) based architecture where they share the convolutional layers amongst agents, sharing thus implicitly knowledge. 
![](https://i.imgur.com/WVMqDLI.gif)


## Methods
### RL
>Reinforcement Learning (RL) allows artificial agents to learn complex tasks by interacting with an environment E using a set of actions A.
The agent learns to take an action a at every step (in a state s) towards the target solution guided by a reward signal r during training. The main goal is to maximize the expected rewards in order to find the optimal policy π∗
![](https://i.imgur.com/aJHZcwr.jpg)

### POMDP ( Partially Observable Markov Decision Process)

### Deep Q-learning
![](https://i.imgur.com/pLQDOi6.jpg)
### MARL
![](https://i.imgur.com/aY3K8x6.jpg)

---

## Result
![](https://i.imgur.com/c2oeB3I.jpg)
![](https://i.imgur.com/SGyEIvK.jpg)








## Reference
Multiple Landmark Detection using Multi-Agent Reinforcement Learning
>Athanasios Vlontzos, Amir Alansary, Konstantinos Kamnitsas, Daniel Rueckert, and Bernhard Kainz
BioMedIA, Computing Dept. Imperial College London
https://github.com/thanosvlo/MARL-for-Anatomical-Landmark-Detection

:::info
**Andrew Yang** @NCTU
:::

###### tags: `Reinforcement Learning` `MARL` 
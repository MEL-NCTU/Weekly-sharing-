# **rlpyt: A Research Code Base for Deep Reinforcement Learning in PyTorch** #

[TOC]

## Introduction
>Since the advent of deep reinforcement learning for game play in 2013  and simulated robotic control shortly after a multitude of new algorithms have flourished. Most are model-free algorithms which can be categorized into three families: deep Q-learning, policy gradients,and Q-value policy gradients.
>We are pleased to share rlpyt, which implements all three algorithm families built on a shared, optimized infrastructure, in a single repository. rlpyt contains modular implementations of many common deep RL algorithms in Python using PyTorch.

## Method
![](https://i.imgur.com/x0gM28z.png)
For sampling, rlpyt offers the following configurations
### Serial
Agent and environments execute within one Python process.
![](https://i.imgur.com/KEJ3iIJ.jpg)
### Parallel-CPU
Agent and environments execute on CPU in parallel worker processes.
![](https://i.imgur.com/bTJjHC6.jpg)
### Alternating-CPU
Environments execute on CPU in parallel workers processes, agent executes in central process
![](https://i.imgur.com/x4oP764.jpg)
### Comparing
When creating or modifying agents, models, algorithms, and environments, serial mode will be the easiest for debugging.

Once that runs smoothly, it is straightforward to explore the more sophisticated infrastructures for parallel sampling, multi-GPU optimization, and asynchronous sampling.

## Result
### Mujoco performance
https://gym.openai.com/envs/#mujoco
Performance for the on-policy algorithms is measured as the average trajectory return across the batch collected at each epoch. Performance for the off-policy algorithms is measured once every 10,000 steps by running the deterministic policy (or, in the case of SAC, the mean policy) without action noise for ten trajectories, and reporting the average return over those test trajectories.
![](https://i.imgur.com/AX7MHFk.png)

![](https://i.imgur.com/CoA58zM.jpg)

compare to https://spinningup.readthedocs.io/zh_CN/latest/spinningup/bench.html
![](https://i.imgur.com/Z6sp4uJ.png)

### Atari performance
By policy gradient algorithms
![](https://i.imgur.com/GLaB02p.png)

By DQN algorithms
![](https://i.imgur.com/U2SIgaV.png)

By rlpyt
![](https://i.imgur.com/bFld1RK.png)

## New data structure : namedarraytuple
rlpyt introduces new object classes "namedarraytuples" for easier organization of collections of
numpy arrays or torch tensors. A namedarraytuple is essentially a namedtuple which exposes indexed
or sliced read/writes into the structure.


![](https://i.imgur.com/wK8iIw4.png)


## Conclusion

>We hope that rlpyt can facilitate adoption of existing deep RL techniques and serve as a launching
point for research into new ones. For example, the more advanced topics of meta-learning, modelbased,
and multi-agent RL are not explicitly addressed in rlpyt, but applicable code components may
still be helpful in accelerating their development. We expect the offering of algorithms to grow over
time as the field matures.

# simple-DVMPC-implemetation

This is a unofficial implementation of Deep Value Model Predictive Control (MPC) algorithm for the paper by F.Farshidian

## Algorithm
---
- Unlike the original paper, this implementation utilized the Cross Entropy Method (CEM) for MPC optimization.

## Environment
---
- 2d navigation environment

![](img/screenshot.png)


## Usage
---
- Train
  - ensemble : if True, train the ensemble model
  - seed : seed number (default: 1234)
  - render : if True, visualize the agent on the environment
  - the default parameters are defined in the `params/value_net.json` and `params/ensemble_value_net.json` file.
   
```
python3 examples/train_deep_value_mpc.py # train the single deep value mpc
```

```
python3 examples/train_deep_value_mpc.py --ensemble # train the ensemble deep value mpc
```

- Test
  - the default load directory is defined in the `params/value_net.json` and `params/ensemble_value_net.json` file.

```
python3 examples/test_deep_value_mpc.py
```

- Visualize
```
python3 examples/visualize_value_net.py
```
  - the start point : (-10, 2)
  - the end point : ( 0, -2)

![](img/value_net_026.png)


## Reference
- [2D vehicle Env](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
- [Deep Value MPC](https://arxiv.org/abs/1910.03358)

# UR10e SAC HER SB3 GYM MuJoCo Reinforcement Learning

Reinforcement learning for teaching a UR10e arm to learn and complete a **reach** task and a **pick and place** task.

This project uses SAC/PPO, curriculum learning, HER, Gymnasium-Robotics, and Stable-Baselines3 within the MuJoCo simulator.

The subject is a **UR10e arm** with a **Robotiq 2F-85 gripper**.


---

## Collaboration

This project was made in collaboration with the **"Jožef Stefan" Institute**.

![Logo of the Institute of "Jožef Stefan".](https://ctop.ijs.si/wp-content/uploads/2018/06/IJS_logo_2-1024x311.jpg)

---
## About
This project was created with the intent of assesing the appropriateness of MuJoCo to be used as a tool for further study of reinforcement learning. Within this project is supplied 2 examples of function: Reach, Pick&Place.
Information about how to use the provided tools are details further within this README aswell as the DEEPDIVE and EXAMPLES. Provided scripts work with the intent of mainly being used with SAC+HER but have minor testing with PPO.
## Workstations

The project used three computers for training, using either CPU or CUDA. CUDA is strongly advised for faster training.

| Computer | CPU | GPU | Mode | RAM | OS |
| -------- | --- | --- | ---- | --- | -- |
| PC 1 | Intel® Core™ i5-7400 × 4 | Intel® HD Graphics 630 | CPU | 8 GiB | Ubuntu 24.04.4 — ROS2 Jazzy |
| PC 2 | AMD Ryzen Threadripper PRO 5975WX | NVIDIA GeForce RTX 4090 | CUDA | 512 GiB | Ubuntu 24.04.4 — ROS2 Humble |
| PC 3 | Intel i5 | NVIDIA RTX 3070 | CUDA | 16 GiB | Windows 10 |

---

## Dependencies
The dependencies included within the table and their version have been used within this project.

| Package | Version |
| ------- | ------- |
| Python | 3.12.3 |
| NumPy | 1.26.4 |
| MuJoCo | 3.8.1 |
| Gymnasium | 1.3.0 |
| Gymnasium-Robotics | 1.4.2 |
| Stable-Baselines3 | 2.9.0 |

---

## Project Files
This chapter defines the main files used within the project
| File | Description |
| ---- | ----------- |
| `ur10e.py` | Custom Gymnasium environment — defines observation, action, reward, and stage logic |
| `ur10e_tester.py` | Training script - CUDA, use to run training simulations |
| `ur10e_tester_cpu.py` | Training script — CPU, use to run training simulations |
| `ur10e_loader.py` | Evaluation script — loads a checkpoint and renders viewport to see results |
| `ur10e_gripper.xml` | MuJoCo XML model of the UR10e arm with Robotiq 2F-85 gripper |
| `ur10e_scene_mod.xml` | MuJoCo XML model of the UR10e arm, base for the ur10e_gripper.xml |
| `2f85.xml` | MuJoCo XML model of the 2f85 gripper from mujoco menagerie |
| `2f85_scene.xml` | MuJoCo XML model of the 2f85 gripper from mujoco menagerie, modified with ground |

---

## Setup

### 1. Clone the repository and install dependencies
```bash
pip install mujoco gymnasium stable-baselines3
```

### 2. Train the model
```bash
# CUDA (recommended)
python ur10e_tester.py

# CPU only
python ur10e_tester_cpu.py
```

### 3. Evaluate a checkpoint
```bash
python ur10e_loader.py
```

---

## Checkpoints

### Where checkpoints are saved

Checkpoints are saved automatically during training to the working directory:

```
ur10e_checkpoint       # SAC checkpoint (saved by ur10e_tester_cpu.py)
ur10e_checkpoint_PPO   # PPO checkpoint (saved by ur10e_tester.py)
ur10e_stage_0          # Final model saved after training completes
```

### How to run a checkpoint

Open `ur10e_loader.py` and ensure the correct checkpoint name is loaded:

```python
model = SAC.load("ur10e_checkpoint", env=env, device="cpu")
```

Then run:
```bash
python ur10e_loader.py
```

### How to continue training from a checkpoint

Load the model before calling `.learn()`:

```python
model = SAC.load("ur10e_checkpoint", env=env, device="cpu")
model.learn(total_timesteps=5_000_000, callback=callback)
```

---

## RL System

### Algorithm: SAC (Soft Actor-Critic)

SAC was chosen at the main algorithm for this research, in combination with HER for improved sample efficiency

| Hyperparameter | Value | Notes |
| -------------- | ----- | ----- |
| `learning_starts` | 32 000 | 32 episodes since doing 32 episodes at once |
| `buffer_size` | 500 000 | Large buffer for better HER learning |
| `batch_size` | 512 | Large batch for stable gradients |
| `gradient_steps` | 1 | One update per environment step |
| `tau` | 0.005 | Soft target network update coefficient |
| `learning_rate` | 1e-3 | Slightly higher than default; works well with HER |
| `gamma` | 0.99 | Discount factor |
| `ent_coef` | auto | Entropy coefficient — tuned automatically |
| `use_sde` | True | State-Dependent Exploration for smoother actions |
| `sde_sample_freq` | 4 | New noise sample every 4 steps |
| HER `n_sampled_goal` | 4 | 4 hindsight goals relabelled per real transition |
| HER `goal_selection_strategy` | future | Goals are sampled from future states in the same episode |

### Algorithm: PPO (Proximal Policy Optimization)

PPO was considered as an alternative algorithm for this research but remained just as a comparison. 

| Hyperparameter | Value | Notes |
| -------------- | ----- | ----- |
| `learning_rate` | 3e-4 | |
| `n_steps` | 1 024 | Steps per environment per update |
| `batch_size` | 64 | Minibatch size for gradient updates |
| `n_epochs` | 10 | Times each rollout batch is reused |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | Generalised Advantage Estimation lambda |
| `clip_range` | 0.2 | PPO clipping parameter |
| `ent_coef` | 0.01 | Entropy coefficient for exploration |
| `vf_coef` | 0.5 | Value function coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `verbose` | 1 | |

---

## Deeper Explanations

*Link to detailed explanation document — WIP*

## Example Results

*Link to results analysis and graphs — WIP*
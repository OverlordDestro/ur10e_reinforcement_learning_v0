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
## P&P observation space
These values describe the observation space for Pick and Place

## Observation — 34 dimensions

| Index | Component | Dims | Description |
| ----- | --------- | ---- | ----------- |
| 0–5 | `cos(q)` | 6 | Cosine of the 6 UR10e arm joint angles |
| 6–11 | `sin(q)` | 6 | Sine of the 6 UR10e arm joint angles |
| 12–17 | Joint velocities | 6 | Angular velocity of each arm joint (`qvel`) |
| 18–20 | TCP Z-axis | 3 | Z-axis column of the TCP rotation matrix (flange pointing direction) |
| 21–23 | Object position | 3 | XYZ world position of `object0` |
| 24–32 | Object rotation | 9 | Full 3×3 rotation matrix of `object0` (row-major flattened) |
| 33 | Verification steps | 1 | Current hold-step counter used for stage transition checks |
| 34 | Stage | 1 | Current curriculum stage (0–3) |

---

## Achieved Goal — 3 dimensions
The achieved goal changes between stages to help guide the agent
| Stage | Value | Description |
| ----- | ----- | ----------- |
| 0–1 | TCP position | XYZ world position of the Tool Center Point (end-effector) |
| 2 | Object position | XYZ world position of `object0` |

---

## Desired Goal — 3 dimensions
The desired goal changes between stages to help guide the agent
| Stage | Value | Description |
| ----- | ----- | ----------- |
| 0–1 | Object position | XYZ world position of `object0` |
| 2 | Lift target position | XYZ world position of `target` |

---

## Action Space — 7 dimensions

| Index | Component | Range | Description |
| ----- | --------- | ----- | ----------- |
| 0 | Cartesian X residual | [−1, 1] | Policy correction for TCP X motion, scaled by 0.08 m |
| 1 | Cartesian Y residual | [−1, 1] | Policy correction for TCP Y motion, scaled by 0.08 m |
| 2 | Cartesian Z residual | [−1, 1] | Policy correction for TCP Z motion, scaled by 0.08 m |
| 3–5 | Unused | [−1, 1] | Reserved padding — ignored by the controller |
| 6 | Gripper signal | [−1, 1] | Open/close command — overridden by stage logic in current implementation |

>3-5 was used for rotation but became obsolete, reserved for any future implementations
---

## 3-Stage Curriculum

The task is broken into three sequential stages. The agent must satisfy a condition at each stage before advancing. Stage transitions are rewarded with a bonus.

| Stage | Name | Gripper | Success Condition | Hold Steps | Stage Bonus |
| ----- | ---- | ------- | ----------------- | ---------- | ----------- |
| 0 | Coarse reach | Open | TCP within 6 cm of object | 4 | +2.0 |
| 1 | Fine reach | Open | TCP within 2 cm of object | 10 | +4.0 |
| 2 | Lift | Closed | Object lifted ≥ 0.2 m above initial Z | 10 | +10.0 |

>original success condition for stage 2 was set to reach the target but was simplified to 0.2m above the ground for concept testing

## target marker colors
For better debugging and observation, the target changes colors to signify the current stage.

| Colour | Meaning |
| ------ | ------- |
| 🔴 Red | Stage 0 — coarse reach |
| 🟠 Orange | Stage 1 — fine reach |
| 🟢 Green | Stage 2 — lifting |
| 🔵 Blue | Episode success |

### Episode Reset

Each episode the arm resets to the `UR10E_home` keyframe. The object is placed at a **random position** within the reachable workspace:

- X: uniform in `[−0.40, +0.40]` m
- Y: uniform in `[+0.6, +0.8]` m
- Z: fixed at `0.025` m (on the table)
- Yaw: random in `[0, 2π]`

The lift target is set to the object's spawn position `+ 0.2 m` in Z.

---

## Reward Design

The total reward per step is:

```
reward = distance_reward + progress_reward + smoothness_penalty + stage_bonus
```

| Component | Formula | Description |
| --------- | ------- | ----------- |
| `distance_reward` | `‖achieved − desired‖` | Dense penalty proportional to distance to goal |
| `progress_reward` | `+3.0 × (prev_dist − curr_dist)` | Reward for closing the distance to the target each step |
| `smoothness_penalty` | `−0.01 × Σ(Δq²)` | Penalises large joint velocity changes to encourage smooth motion |
| `stage_bonus` | `+2 / +4 / +4 / +10` | One-off bonus awarded when each stage transition is triggered |

---

## Success Conditions

| Stage | Condition |
| ----- | --------- |
| Stage 0 → 1 | TCP distance to object < **6 cm** with gripper open |
| Stage 1 → 2 | TCP distance to object < **2 cm**  with gripper open |
| Episode success | Object lifted ≥ **0.2 m** Z |

> Early termination is enabled by default — the episode ends immediately on success

---

## Deeper Explanations

*Link to detailed explanation document — WIP*

## Example Results

*Link to results analysis and graphs — WIP*
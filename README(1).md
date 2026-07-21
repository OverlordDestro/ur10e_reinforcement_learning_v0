# UR10e Reinforcement Learning in MuJoCo

This repository contains a reinforcement-learning project for controlling a simulated **Universal Robots UR10e** robot arm with a **Robotiq 2F-85** gripper in MuJoCo. The project contains two Gymnasium environments:

- **Reach**: the Tool Center Point (TCP) learns to move to a randomly placed target.
- **Pick and Place (PGP)**: the TCP reaches an object, closes the gripper, and lifts the object.

The training and evaluation scripts are built around **Stable-Baselines3**, mainly **SAC** and **PPO**, with support for HER-compatible `achieved_goal` and `desired_goal` observations.

---

## Repository layout

| File | Purpose |
| --- | --- |
| `README.md` | Main project overview and operation guide. |
| `README_REACH.md` | Detailed documentation for the reach environment. |
| `README_PGP.md` | Detailed documentation for the pick-and-place environment. |
| `ur10e_reach.py` | Gymnasium MuJoCo environment for the reach task. Registers `UR10E-reach-v0`. |
| `ur10e_pgp.py` | Gymnasium MuJoCo environment for the pick-and-place task. Registers `UR10E-pgp-v0`. |
| `ur10e_tester.py` | CUDA/GPU training script. Uses vectorized environments and saves checkpoints/plots. |
| `ur10e_tester_cpu.py` | CPU training script. Uses one environment and can render live simulation. |
| `ur10e_loader.py` | Evaluation/replay script. Loads a trained SAC/PPO model and renders it. |
| `ur10e_gripper.xml` | MuJoCo XML scene containing the UR10e arm, Robotiq 2F-85 gripper, object, and target. |
| `mesh/ur10e/` | UR10e mesh files required by the XML model. |
| `mesh/robotiq_2f85/` | Robotiq 2F-85 mesh files required by the XML model. |

The Python imports expect the environment files to be named exactly:

```text
ur10e_reach.py
ur10e_pgp.py
```

If your files have names like `ur10e_reach(2).py` or `ur10e_pgp(2).py`, rename them before running the project.

---

## Required software

The project was tested with the following versions:

| Package | Version used |
| --- | --- |
| Python | 3.12.3 |
| NumPy | 1.26.4 |
| MuJoCo | 3.8.1 |
| Gymnasium | 1.3.0 |
| Gymnasium-Robotics | 1.4.2 |
| Stable-Baselines3 | 2.9.0 |
| PyTorch | Depends on CPU/CUDA setup |

Install the main dependencies with:

```bash
pip install numpy mujoco gymnasium gymnasium-robotics stable-baselines3 matplotlib torch
```

For CUDA training, install a CUDA-enabled PyTorch build that matches your GPU and driver.

---

## Hardware notes

The project can run on CPU or CUDA, but MuJoCo simulation and RL training are much faster on an NVIDIA CUDA GPU.

| Mode | Recommended use |
| --- | --- |
| CPU | Debugging, rendering, small tests, or systems without CUDA. |
| CUDA | Main training runs, especially with many parallel environments. |

The CUDA script is designed for fast training and normally uses `render_mode=None`. The CPU script can use `render_mode="human"` for real-time viewing, but this slows training heavily.

---

## Environment IDs

Importing the environment files automatically registers the environments with Gymnasium.

| Environment ID | File | Task |
| --- | --- | --- |
| `UR10E-reach-v0` | `ur10e_reach.py` | Reach a random target with the TCP. |
| `UR10E-pgp-v0` | `ur10e_pgp.py` | Reach, grasp, and lift an object. |

The training and loader scripts import both environment files:

```python
import ur10e_pgp
import ur10e_reach
```

This is required because Gymnasium only knows the custom environments after the modules have been imported.

---

## Basic workflow

### 1. Choose the task

Open `ur10e_tester.py`, `ur10e_tester_cpu.py`, or `ur10e_loader.py` and set:

```python
ENVIRONMENT = "UR10E-reach-v0"  # reach task
```

or:

```python
ENVIRONMENT = "UR10E-pgp-v0"    # pick-and-place task
```

### 2. Choose the algorithm

Set:

```python
ALGORITHM = "SAC"
```

or:

```python
ALGORITHM = "PPO"
```

SAC is the main recommended algorithm for this project. PPO is included for comparison but may require more careful evaluation settings.

### 3. Choose CPU or CUDA training

For CUDA training:

```bash
python ur10e_tester.py
```

For CPU training:

```bash
python ur10e_tester_cpu.py
```

### 4. Evaluate a trained checkpoint

Edit these variables in `ur10e_loader.py`:

```python
ALGORITHM = "SAC"
MODEL_PATH = "ur10e_pgp_SAC"
ENVIRONMENT = "UR10E-pgp-v0"
DETERMINISTIC = True
```

Then run:

```bash
python ur10e_loader.py
```

The loader renders the environment, prints per-episode information, tracks success ratio, and saves an evaluation graph as:

```text
eval_rewards.png
```

---

## Important variables in the training scripts

| Variable | Meaning |
| --- | --- |
| `ALGORITHM` | Selects `SAC` or `PPO`. |
| `MAX_EPISODE_STEPS` | Maximum number of environment steps per episode. The environments are registered with 1000 steps. |
| `EPISODES` | Used to calculate `TOTAL_TIMESTEPS`. It is not always the exact number of completed episodes because early success can end an episode before the step limit. |
| `TOTAL_TIMESTEPS` | Total training budget, calculated as `MAX_EPISODE_STEPS * EPISODES`. |
| `LOG_EVERY` | Episode interval for logging mean reward. |
| `CHECKPOINT_EVERY` | Episode interval for saving checkpoints. |
| `TRAIN_DEVICE` | `cuda` or `cpu`. The CUDA script rejects CPU use and the CPU script rejects CUDA use. |
| `ENVIRONMENT` | Must match the selected Gymnasium environment ID. |
| `MODEL_SAVE` | Final model filename. |
| `CHECKPOINT_SAVE` | Intermediate checkpoint filename. |
| `N_ENVIRONMENTS` | Number of parallel environments. CUDA script uses more, CPU script uses one. |
| `LOG_DIR` | Folder used by Stable-Baselines3 monitor logs. |

---

## Important variables in the loader

| Variable | Meaning |
| --- | --- |
| `ALGORITHM` | Which class to use when loading the model: `SAC.load(...)` or `PPO.load(...)`. |
| `MODEL_PATH` | Name/path of the checkpoint or final model to load. |
| `ENVIRONMENT` | Environment ID to evaluate. Must match the model that was trained. |
| `NO_EARLY_TERMINATION` | If `True`, the environment continues even after success, useful for viewing behavior. |
| `MAX_EPISODE_STEPS` | Maximum replay length. The loader comments note that CPU rendering may appear to use different effective timing. |
| `EPISODES` | Number of evaluation episodes. |
| `DETERMINISTIC` | Whether to use deterministic actions. SAC usually works well deterministic; PPO may behave worse if forced deterministic. |
| `SUCCESS_LOG_EVERY` | How often success ratio is logged/plotted. |

---

## Training outputs

Training scripts save:

| Output | Description |
| --- | --- |
| `MODEL_SAVE` | Final trained model. The actual filename is whatever `MODEL_SAVE` is set to. |
| `CHECKPOINT_SAVE` | Periodic checkpoint saved every `CHECKPOINT_EVERY` episodes. |
| `training_reward_total.png` | Plot of total episode reward over training. |
| `training_reward_components.png` | Reward component and success ratio plots. |
| `logs/` | Stable-Baselines3 monitor logs. |

Evaluation saves:

| Output | Description |
| --- | --- |
| `eval_rewards.png` | Evaluation plot with reward components and success ratio. |

---

## Model and scene file

`ur10e_gripper.xml` defines the simulated scene. It contains:

- UR10e arm joints and actuators.
- Robotiq 2F-85 gripper.
- A TCP site named `UR10E_TCP`.
- A movable object body named `object0`.
- A visual target body named `target` with target geom `target_geom`.
- A home keyframe named `UR10E_home`.

The XML expects the mesh folders to be available relative to the XML file:

```text
mesh/ur10e/
mesh/robotiq_2f85/
```

If MuJoCo cannot load the XML, first check that these mesh folders exist and that the script is being run from the correct project directory.

---

## Controller overview

Both environments use a Jacobian-based Cartesian controller:

1. The policy outputs an action in the range `[-1, 1]`.
2. The action is scaled into a Cartesian TCP command.
3. `mujoco.mj_jacSite(...)` computes the TCP Jacobian.
4. A damped least-squares inverse kinematics step converts TCP motion into joint deltas.
5. Joint deltas are clipped by `max_joint_delta`.
6. The first six MuJoCo controls command the UR10e joint targets.
7. The seventh control commands the gripper.

Important tuning values:

| Variable | Meaning |
| --- | --- |
| `cartesian_action_scale` | Maximum TCP motion requested by the policy per step. Example: `0.16` means up to 16 cm per step before IK conversion. |
| `max_joint_delta` | Maximum allowed joint target change per control update. |
| `rotation_gain` | Scale for the orientation correction or rotation command. |
| `damping` | Damped least-squares stability term. Larger values are more stable but less responsive. |
| `gamma` | Discount factor used in potential-based shaping. |

If the arm shakes near the goal, reduce `cartesian_action_scale`, `max_joint_delta`, or `rotation_gain`.

---

## Reward components

The training and loader scripts expect the environments to return these reward components inside `info["reward_components"]`:

| Component | Meaning |
| --- | --- |
| `distance_reward` | Negative distance between achieved goal and desired goal. |
| `potential_reward` | Potential-based shaping reward based on distance improvement. |
| `orientation_reward` | Reward for keeping or achieving the desired TCP orientation. |
| `total_reward` | Sum used as the environment reward before any additional success/stage bonus. |

The scripts also read:

```python
info.get("is_success", False)
```

for success-ratio tracking.

---

## Common operation mistakes

### Environment and model do not match

A model trained on `UR10E-reach-v0` should be evaluated with `UR10E-reach-v0`. A model trained on `UR10E-pgp-v0` should be evaluated with `UR10E-pgp-v0`.

### Wrong algorithm loader

If the model was trained with SAC, load it with:

```python
model = SAC.load(MODEL_PATH, env=env, device="cpu")
```

If it was trained with PPO, load it with:

```python
model = PPO.load(MODEL_PATH, env=env, device="cpu")
```

### PPO deterministic evaluation

The loader has a `DETERMINISTIC` flag. In this project, SAC can usually be viewed deterministically. PPO may behave badly if deterministic evaluation is forced, so use:

```python
DETERMINISTIC = False
```

when evaluating PPO if the replay does not match training performance.

### CUDA rendering

The loader uses `render_mode="human"`, so it is meant to run on CPU. CUDA is useful for training, but real-time rendering should generally be done with CPU loading.

---

## File-specific documentation

For task-specific details, read:

- `README_REACH.md` for the reach task.
- `README_PGP.md` for the pick-and-place task.

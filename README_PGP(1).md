# UR10e Pick-and-Place Environment

This document explains the pick-and-place environment implemented in `ur10e_pgp.py`.

The task uses a simulated UR10e arm with a Robotiq 2F-85 gripper. The agent must move the TCP to the object, close the gripper, and lift the object. The environment uses staged progression to make the problem easier to learn.

---

## Environment registration


```python
register(
    id="UR10E-pgp-v0",
    entry_point="ur10e_pgp:ur10eEnv",
    max_episode_steps=1000,
)
```

Use it with:

```python
import ur10e_pgp
import gymnasium as gym

env = gym.make("UR10E-pgp-v0", render_mode="human", max_episode_steps=1000)
```

---

## Task objective

The pick-and-place task is divided into stages:

```text
Stage 0: Move TCP near the object.
Stage 1: Move TCP very close to the object.
Stage 2: Close the gripper and lift the object.
```

The task is considered successful when the object is lifted high enough while the gripper is closed.

---

## Reset behavior

At the start of every episode:

1. The robot resets to the `UR10E_home` keyframe.
2. The gripper state is reset.
3. Stage is reset to `0`.
4. The object is randomly placed on the table.
5. The target marker is placed above the object.
6. Object velocity is reset to zero.
7. MuJoCo state is updated using `mujoco.mj_forward(...)`.

Object placement:

```python
x = uniform(-0.4, 0.4)
y = uniform(0.6, 0.8)
z = 0.025
```

Target placement:

```python
target_x = object_x
target_y = object_y
target_z = 0.1
```

The current success condition in the uploaded file uses object height directly:

```python
gripper_close and object0_pos[2] > 0.2
```

So the marker is mainly visual/guidance, while final success is based on lifting the object above `0.2 m`.

---

## Observation space

The observation dictionary contains:

```python
spaces.Dict({
    "observation": Box(shape=(34,), dtype=np.float64),
    "achieved_goal": Box(shape=(3,), dtype=np.float64),
    "desired_goal": Box(shape=(3,), dtype=np.float64),
})
```

### `observation` vector — 34 dimensions

| Index | Component | Dimensions | Description |
| --- | --- | ---: | --- |
| `0–5` | `cos(q)` | 6 | Cosine of the six UR10e joint angles. |
| `6–11` | `sin(q)` | 6 | Sine of the six UR10e joint angles. |
| `12–17` | `qvel` | 6 | Joint velocities of the six arm joints. |
| `18–20` | TCP Z-axis | 3 | Third column of TCP rotation matrix. Used to know whether TCP points downward. |
| `21–23` | Object position | 3 | World XYZ position of `object0`. |
| `24–32` | Object rotation | 9 | Flattened 3×3 rotation matrix of `object0`. |
| `33` | Stage | 1 | Current curriculum stage. |

### `achieved_goal` and `desired_goal`

The goal fields change depending on the stage:

| Stage | `achieved_goal` | `desired_goal` | Meaning |
| --- | --- | --- | --- |
| `0` | TCP position | Object position | Coarse reach toward object. |
| `1` | TCP position | Object position | Fine reach toward object. |
| `2` | Object position | Target position | Lift object toward target/height goal. |

This structure is useful for HER because the environment exposes the current achieved and desired goal for each stage.

---

## Action space

The environment uses a 7-dimensional action space:

```python
self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
```

| Action index | Meaning | Used? |
| --- | --- | --- |
| `0` | TCP X movement command | Yes |
| `1` | TCP Y movement command | Yes |
| `2` | TCP Z movement command | Yes |
| `3` | TCP rotation X command | No, ignored by current PGP controller |
| `4` | TCP rotation Y command | No, ignored by current PGP controller |
| `5` | TCP rotation Z command | No, ignored by current PGP controller |
| `6` | Gripper open/close command | Yes |

The PGP script keeps the 7D action space because the full robot has six arm joints plus one gripper actuator, and because older scripts/models expect 7 action values.

---

## TCP orientation behavior

In `ur10e_pgp.py`, TCP orientation is hard-coded toward world down.

The controller computes:

```python
current_z = TCP local Z-axis
desired_z = [0, 0, -1]
rot_error = cross(current_z, desired_z)
rot_error *= rotation_gain
```

That means the policy does **not** directly control rotation with `action[3:6]` in this PGP file. Rotation is automatically corrected so the gripper points downward toward the object/table.

---

## Controller pipeline

Each environment step does the following:

1. Validate the action shape. PGP expects exactly `(7,)`.
2. Use `action[:3]` as Cartesian TCP movement.
3. Use `action[6]` as gripper signal.
4. Compute TCP orientation error so the TCP points downward.
5. Combine position and rotation command into a 6D task-space command.
6. Use `mujoco.mj_jacSite(...)` to compute the TCP Jacobian.
7. Use damped least-squares IK to compute joint deltas.
8. Clip joint deltas with `max_joint_delta`.
9. Clip desired joint positions to joint limits.
10. Send the first six `ctrl` values to the UR10e arm.
11. Send `ctrl[6]` to the gripper actuator.
12. Run MuJoCo simulation with `do_simulation(...)`.

Important controller parameters:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `cartesian_action_scale` | `0.16` | Max TCP translation command per step. |
| `max_joint_delta` | `1.0` | Max joint target change per control update. |
| `rotation_gain` | `0.5` | Strength of automatic TCP-down orientation correction. |
| `damping` | `0.05` | Damped least-squares stabilization. |
| `gamma` | `0.99` | Discount factor for potential reward. |

---

## Stage logic

### Stage 0 — coarse reach

Goal:

```text
Move TCP close to object while gripper is open.
```

Condition:

```python
distance(TCP, object) < 0.06
and gripper_state < 0.3
```

Effect:

```python
stage = 1
reward += 5.0
```

---

### Stage 1 — fine reach

Goal:

```text
Move TCP very close to object while gripper is still open.
```

Condition:

```python
distance(TCP, object) < 0.02
and gripper_state < 0.3
```

Effect:

```python
stage = 2
reward += 5.0
```

---

### Stage 2 — lift

Goal:

```text
Close the gripper and lift the object.
```

Condition:

```python
gripper_state > 0.4
and object0_pos[2] > 0.2
```

Effect:

```python
episode_success = True
terminated = True
reward += 10.0
```

If `no_early_termination=True`, the episode is marked successful but does not immediately stop. This is useful for replaying and observing what happens after success.

---

## Gripper state

The gripper command is mapped from policy range `[-1, 1]` into MuJoCo actuator range `[0, 255]`:

```python
ctrl[6] = clip((gripper_signal + 1.0) * 0.5 * 255.0, 0.0, 255.0)
```

The measured normalized gripper state is:

```python
gripper_state = qpos[left_driver_joint] / 0.9
```

Thresholds used by the task:

| Name | Condition | Meaning |
| --- | --- | --- |
| `gripper_open` | `gripper_state < 0.3` | Gripper is open enough for reaching stage. |
| `gripper_close` | `gripper_state > 0.4` | Gripper is closed enough for lifting success. |

There is a small dead zone between `0.3` and `0.4` to avoid ambiguous half-open/half-closed states.

---

## Reward function

The environment returns reward components inside `info["reward_components"]`:

| Component | Meaning |
| --- | --- |
| `distance_reward` | Negative distance between achieved goal and desired goal. |
| `potential_reward` | Reward for making progress compared with previous distance. |
| `orientation_reward` | Reward/penalty based on TCP downward alignment. |
| `total_reward` | Combined reward before extra stage/success bonuses. |

Formula overview:

```text
distance_reward = -distance
potential_reward = gamma * (-current_distance) - (-previous_distance)
orientation_reward = z_dot * 0.5 - 0.5
reward = distance_reward + potential_reward + orientation_reward
```

Stage bonuses are then added in `step()`:

| Event | Bonus |
| --- | ---: |
| Stage 0 -> Stage 1 | `+5.0` |
| Stage 1 -> Stage 2 | `+5.0` |
| Final lift success | `+10.0` |

---

## Rendering colors

When `render_mode="human"`, the target color changes according to stage:

| Color | Meaning |
| --- | --- |
| Red | Stage 0: coarse reach. |
| Orange | Stage 1: fine reach. |
| Green | Stage 2: lift. |
| Blue | Episode success. |

---

## How to train pick-and-place

In `ur10e_tester.py` or `ur10e_tester_cpu.py`, set:

```python
ENVIRONMENT = "UR10E-pgp-v0"
MODEL_SAVE = "ur10e_pgp_SAC"
CHECKPOINT_SAVE = "ur10e_pgp_checkpoint"
ALGORITHM = "SAC"
```

Then run:

```bash
# CUDA training
python ur10e_tester.py

# CPU training
python ur10e_tester_cpu.py
```

For CUDA training, the uploaded script uses multiple vectorized environments. For CPU training, it uses one rendered environment.

---

## How to evaluate pick-and-place

In `ur10e_loader.py`, set:

```python
ALGORITHM = "SAC"
MODEL_PATH = "ur10e_pgp_SAC"
ENVIRONMENT = "UR10E-pgp-v0"
NO_EARLY_TERMINATION = True
DETERMINISTIC = True
```

Then run:

```bash
python ur10e_loader.py
```

The loader prints episode reward, success rate, current stage, distance to current goal, and distance to the box. It also saves:

```text
eval_rewards.png
```

---

## Tuning notes

### The robot reaches the object but fails to lift

Common causes:

- Stage 1 is too strict, so it rarely reaches the gripper-closing stage.
- The gripper closes too late or from a bad angle.
- The object is contacted from the side rather than centered.
- The lift success height is too strict.

Possible changes:

```python
# Easier fine reach
stage1_distance = 0.03  # instead of 0.02

# Easier lift
object0_pos[2] > 0.19  # instead of 0.20
```

If you relax only Z while keeping XY strict, split distance into separate XY and Z checks instead of using one 3D distance.

### The robot shakes near the object

Lower:

```python
self.cartesian_action_scale = 0.04
self.max_joint_delta = 0.25
self.rotation_gain = 0.2
```

### The robot reaches but keeps opening the gripper

Because stage 0 and stage 1 require `gripper_open`, the policy may learn to keep the gripper open while approaching. It must then learn to close only after reaching stage 2. If it struggles, add a stronger stage 2 gripper-closing reward or make the stage transition more obvious.

### Success ratio is lower in loader than trainer

Check these first:

- `ENVIRONMENT` matches the training environment.
- `ALGORITHM` matches the saved model type.
- `MODEL_PATH` points to the correct checkpoint.
- `DETERMINISTIC` is appropriate. PPO may need `DETERMINISTIC = False`.
- `NO_EARLY_TERMINATION` changes viewing behavior but should still preserve `info["is_success"]`.

---

## Relationship to the reach environment

`ur10e_reach.py` is simpler and should be used first to verify that:

- The XML loads correctly.
- The robot can move with the Jacobian controller.
- The training scripts work.
- The loader can replay checkpoints.

After reach is stable, use `ur10e_pgp.py` for the more difficult pick-and-place task.

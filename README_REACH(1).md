# UR10e Reach Environment

This document explains the reach-only environment implemented in `ur10e_reach.py`.

The reach task trains the UR10e TCP to move to a randomly placed target in the MuJoCo scene. The object and gripper remain in the scene for compatibility with the shared XML model, but the task success depends only on the TCP reaching the target.

---

## Environment registration

The file registers the environment as:

```python
register(
    id="UR10E-reach-v0",
    entry_point="ur10e_reach:ur10eEnv",
    max_episode_steps=1000,
)
```

Use it with:

```python
import ur10e_reach
import gymnasium as gym

env = gym.make("UR10E-reach-v0", render_mode="human", max_episode_steps=1000)
```

---

## Task objective

The goal is:

```text
Move the UR10e TCP to the random target position.
```

The environment uses Gymnasium goal observations:

| Goal field | Value |
| --- | --- |
| `achieved_goal` | Current TCP world position. |
| `desired_goal` | Current target world position. |

Success occurs when:

```text
distance(TCP, target) < success_distance
```

and, if enabled:

```text
TCP Z-axis points downward enough: z_dot > success_z_dot
```

The default values are:

```python
self.success_distance = 0.05
self.require_downward_tcp = True
self.success_z_dot = 0.96
```

`z_dot = 1.0` means the TCP Z-axis points perfectly toward world down `[0, 0, -1]`.

---

## Random target generation

At reset, the target is placed randomly in a donut-shaped reachable area around the robot:

```python
xy = uniform(-0.8, 0.8)
z = uniform(0.2, 0.7)
0.7 < norm(xy) < 1.0
```

This means:

- The target is not too close to the robot base.
- The target is not too far outside the workspace.
- The target height is between `0.2 m` and `0.7 m`.

The object is not used for success in this task. It is placed at a fixed position so that the same XML model can be reused.

---

## Observation space

The observation dictionary contains:

```python
spaces.Dict({
    "observation": Box(shape=(33,), dtype=np.float64),
    "achieved_goal": Box(shape=(3,), dtype=np.float64),
    "desired_goal": Box(shape=(3,), dtype=np.float64),
})
```

### `observation` vector — 33 dimensions

| Index | Component | Dimensions | Description |
| --- | --- | ---: | --- |
| `0–5` | `cos(q)` | 6 | Cosine of the six UR10e joint angles. |
| `6–11` | `sin(q)` | 6 | Sine of the six UR10e joint angles. |
| `12–17` | `qvel` | 6 | Joint velocities of the six UR10e arm joints. |
| `18–20` | TCP Z-axis | 3 | Third column of the TCP rotation matrix. Shows TCP pointing direction. |
| `21–23` | Object position | 3 | Position of `object0`. Kept for compatibility/debugging, not used for reach success. |
| `24–32` | Object rotation | 9 | Flattened 3×3 object rotation matrix. Kept for compatibility/debugging. |

### Why use `cos(q)` and `sin(q)`?

Joint angles wrap around at `±π`. Encoding angles as both sine and cosine avoids a discontinuity where `π` and `-π` are physically close but numerically far apart.

---

## Action space

The environment uses a 7-dimensional action space:

```python
self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
```

| Action index | Meaning | Used in reach? |
| --- | --- | --- |
| `0` | TCP X movement command | Yes |
| `1` | TCP Y movement command | Yes |
| `2` | TCP Z movement command | Yes |
| `3` | TCP rotation X command | Yes |
| `4` | TCP rotation Y command | Yes |
| `5` | TCP rotation Z command | Yes |
| `6` | Gripper open/close command | Ignored/kept open |

The gripper is kept open in the reach task because the task only requires reaching the target.

---

## Action parsing

`_parse_action()` validates and scales the action:

```python
cart_action = action[:3]
rot_action = action[3:6]
cart_action = clip(cart_action, -1, 1) * cartesian_action_scale
rot_action = clip(rot_action, -1, 1) * rotation_gain
```

So the policy outputs normalized values, and the environment converts them into physical movement commands.

Important values:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `cartesian_action_scale` | `0.16` | Maximum TCP translation command per step. |
| `rotation_gain` | `0.5` | Maximum scaled rotation residual. |
| `max_joint_delta` | `1.0` | Maximum joint target change after IK. |
| `damping` | `0.05` | Damped least-squares stabilization value. |

---

## TCP orientation

In this reach file, the policy controls rotation using `action[3:6]`.

The environment still rewards and checks whether the TCP points downward, but the downward orientation is **not directly hard-coded into the controller** in this file. Instead:

- The policy provides the rotation command.
- The reward includes `orientation_reward` based on `z_dot`.
- Success can require `z_dot > 0.96` if `require_downward_tcp = True`.

This is different from the pick-and-place file, where the rotation error is computed directly from the current TCP Z-axis toward `[0, 0, -1]`.

---

## Controller pipeline

Each step works like this:

1. Read and scale the action.
2. Store previous TCP-target distance for potential-based shaping.
3. Build a 6D task command:

```python
target_delta_6d = [cartesian_xyz, rotation_xyz]
```

4. Compute the TCP Jacobian with:

```python
mujoco.mj_jacSite(..., site="UR10E_TCP")
```

5. Stack position and rotation Jacobians into a 6D Jacobian.
6. Use damped least-squares IK to compute joint deltas.
7. Clip joint deltas using `max_joint_delta`.
8. Send the first six control values to the UR10e joint actuators.
9. Keep the gripper open.
10. Simulate with `self.do_simulation(ctrl, self.frame_skip)`.

---

## Reward function

The reward is composed of:

| Component | Meaning |
| --- | --- |
| `distance_reward` | Negative TCP-target distance. |
| `potential_reward` | Extra reward for moving closer than the previous step. |
| `orientation_reward` | Small reward/penalty based on how downward the TCP points. |
| `total_reward` | Sum of the above components. |

Formula overview:

```text
distance_reward = -distance
potential_reward = gamma * (-current_distance) - (-previous_distance)
orientation_reward = z_dot * 0.5 - 0.5
reward = distance_reward + potential_reward + orientation_reward
```

On success, the environment adds an additional `+10.0` reward bonus.

---

## Success and termination

At each step:

```python
orientation_ok = True
if require_downward_tcp:
    orientation_ok = z_dot > success_z_dot

if distance < success_distance and orientation_ok:
    episode_success = True
    terminated = True
```

If `no_early_termination=True`, success is still marked, but the episode does not immediately terminate. This is useful for viewing behavior in the loader.

---

## Rendering colors

When `render_mode="human"`, the target color changes:

| Color | Meaning |
| --- | --- |
| Red | Target has not been reached yet. |
| Blue | Episode success. |

---

## How to train reach

In `ur10e_tester.py` or `ur10e_tester_cpu.py`, set:

```python
ENVIRONMENT = "UR10E-reach-v0"
MODEL_SAVE = "ur10e_reach_SAC"
CHECKPOINT_SAVE = "ur10e_reach_checkpoint"
ALGORITHM = "SAC"
```

Then run the correct training script:

```bash
# CUDA training
python ur10e_tester.py

# CPU training
python ur10e_tester_cpu.py
```

---

## How to evaluate reach

In `ur10e_loader.py`, set:

```python
ALGORITHM = "SAC"
MODEL_PATH = "ur10e_reach_SAC"
ENVIRONMENT = "UR10E-reach-v0"
NO_EARLY_TERMINATION = True
DETERMINISTIC = True
```

Then run:

```bash
python ur10e_loader.py
```

---

## Tuning notes

### The robot reaches the target but shakes

Try lowering:

```python
self.cartesian_action_scale = 0.04
self.max_joint_delta = 0.25
self.rotation_gain = 0.2
```

Lower values are slower but more stable. Higher values are faster but can overshoot and shake near the target.

### The robot moves too slowly

Increase `cartesian_action_scale` gradually:

```python
0.04 -> 0.08 -> 0.16
```

Avoid jumping directly to very large values unless the policy has enough time to learn stable control.

### The robot reaches position but fails success

Check `require_downward_tcp` and `success_z_dot`. If the TCP reaches the target but is not pointing down enough, success will not trigger.

To ignore orientation during success:

```python
self.require_downward_tcp = False
```

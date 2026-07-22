# Pick and Place

## About
This file details a bit about the specifics of reach file

---

## Reach observation space
These values describe the observation space for Reach

## Observation — 33 dimensions

| Index | Component | Dims | Description |
| ----- | --------- | ---- | ----------- |
| 0–5 | `cos(q)` | 6 | Cosine of the 6 UR10e arm joint angles |
| 6–11 | `sin(q)` | 6 | Sine of the 6 UR10e arm joint angles |
| 12–17 | Joint velocities | 6 | Angular velocity of each arm joint (`qvel`) |
| 18–20 | TCP Z-axis | 3 | Z-axis column of the TCP rotation matrix (flange pointing direction) |
| 21–23 | Object position | 3 | XYZ world position of `object0` |
| 24–32 | Object rotation | 9 | Full 3×3 rotation matrix of `object0` (row-major flattened) |


## Achieved Goal — 1 dimension
The achieved goal changes between stages to help guide the agent
| Stage | Value | Description |
| ----- | ----- | ----------- |
| 0 | TCP position | XYZ world position of the Tool Center Point (end-effector) |


## Desired Goal — 1 dimension
The desired goal changes between stages to help guide the agent
| Stage | Value | Description |
| ----- | ----- | ----------- |
| 0 | Target position | XYZ world position of `target` |

## Action Space — 7 dimensions

| Index | Component | Range | Description |
| ----- | --------- | ----- | ----------- |
| 0 | Cartesian X residual | [−1, 1] | Policy correction for TCP X motion, scaled by 0.16 m |
| 1 | Cartesian Y residual | [−1, 1] | Policy correction for TCP Y motion, scaled by 0.16 m |
| 2 | Cartesian Z residual | [−1, 1] | Policy correction for TCP Z motion, scaled by 0.16 m |
| 3–5 | Unused | [−1, 1] | Reserved padding — ignored by the controller |
| 6 | Gripper signal | [0, 1] | how open and closed the gripper is 0 open / 1 closed|

>3-5 was used for rotation but became obsolete, reserved for any future implementations
---
## target marker colors
For better debugging and observation, the target changes colors to signify the current stage.

| Colour | Meaning |
| ------ | ------- |
| 🔴 Red | Stage 0 — reach |
| 🔵 Blue | Episode success |

### Episode Reset

Each episode the arm resets to the `UR10E_home` keyframe. The target is placed at a **random position** in a donut area around the robot:

- X: uniform in `[−0.80, +0.80]` m
- Y: uniform in `[+0.80, +0.80]` m
- Z: uniform in `[+0.20, +0.70]` m
- Yaw: random in `[0, 2π]`

---

## Reward Design

The total reward per step is:

```
reward = base_reward + potential_reward + orientation_reward
```

| Component | Formula | Description |
| --------- | ------- | ----------- |
| `distance_reward` | `-‖achieved − desired‖` | Dense penalty proportional to distance to goal |
| `potential_reward` | `+ γ × (prev_dist − curr_dist)` | Reward for being closer this step than previous step |
| `orientation_reward` | `0.5 × z_dot − 0.5` | Reward for having the TCP oriented downwards, -0.5 for normalization |
| `stage_bonus` | `+10` | One-off bonus awarded when success is triggered |

---

## Success Condition

distance < 0.05m and orientation downwards
> Early termination is enabled by default — the episode ends immediately on success

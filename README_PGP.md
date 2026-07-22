# Pick and Place

## About
This file details a bit about the specifics of pick and place file

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
| 33 | Stage | 1 | Current curriculum stage (0–2) |


## Achieved Goal — 3 dimensions
The achieved goal changes between stages to help guide the agent
| Stage | Value | Description |
| ----- | ----- | ----------- |
| 0–1 | TCP position | XYZ world position of the Tool Center Point (end-effector) |
| 2 | Object position | XYZ world position of `object0` |

## Desired Goal — 3 dimensions
The desired goal changes between stages to help guide the agent
| Stage | Value | Description |
| ----- | ----- | ----------- |
| 0–1 | Object position | XYZ world position of `object0` |
| 2 | Lift target position | XYZ world position of `target` |

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

## 3-Stage Curriculum

The task is broken into three sequential stages. The agent must satisfy a condition at each stage before advancing. Stage transitions are rewarded with a bonus.

| Stage | Name | Gripper | Success Condition | Stage Bonus |
| ----- | ---- | ------- | ----------------- |  ----------- |
| 0 | Coarse reach | Open | TCP within 6 cm of object | +5.0 |
| 1 | Fine reach | Open | TCP within 2 cm of object | +5.0 |
| 2 | Lift | Closed | Object lifted ≥ 0.2 m above initial Z | +10.0 |

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
reward = base_reward + potential_reward + orientation_reward
```

| Component | Formula | Description |
| --------- | ------- | ----------- |
| `distance_reward` | `-‖achieved − desired‖` | Dense penalty proportional to distance to goal |
| `potential_reward` | `+ γ × (prev_dist − curr_dist)` | Reward for being closer this step than previous step |
| `orientation_reward` | `0.5 × z_dot − 0.5` | Reward for having the TCP oriented downwards, -0.5 for normalization |
| `stage_bonus` | ` +5 / +5 / +10` | One-off bonus awarded when each stage transition is triggered |

---

## Success Conditions

| Stage | Condition |
| ----- | --------- |
| Stage 0 → 1 | TCP distance to object < **6 cm** with gripper open |
| Stage 1 → 2 | TCP distance to object < **2 cm**  with gripper open |
| Episode success | Object lifted ≥ **0.2 m** Z |

> Early termination is enabled by default — the episode ends immediately on success

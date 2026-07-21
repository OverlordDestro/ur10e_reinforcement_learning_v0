# Evaluation script
# Loads a trained checkpoint,
# runs deterministic or stochastic evaluation
# and plots evaluation metrics.

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
import sys
import os
import torch
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import ur10e_pgp
import ur10e_reach
print("environment file imported")


#variables for which algorithm and agent you want to use
ALGORITHM = "SAC"      # Change to "PPO" when needed
MODEL_PATH = "ur10e_pgp_SAC" #change to the trained model checkpoint
ENVIRONMENT = "UR10E-pgp-v0"  # Change to "UR10E-reach-v0" when needed
NO_EARLY_TERMINATION = True  # Should the episode terminate on success or keep playing?
MAX_EPISODE_STEPS = 500  # Maximum steps per episode, for CPU leave it at half your intended timesteps, for GPU set it to your intended timesteps
                        #for some unknown reason, CPU uses 2x the set timesteps while the GPU uses the set timesteps, why? idk?
EPISODES = 30           #how many episodes to run
DETERMINISTIC = True  # Should the agent act deterministically or stochastically? (SAC can be deterministic, PPO must be stochastic)
SUCCESS_LOG_EVERY = 10  # Plot success ratio every N episodes
NUM_STEPS = MAX_EPISODE_STEPS * EPISODES  # Total number of steps to run the evaluation for, EPISODES * MAX_EPISODE_STEPS = NUM_STEPS


print("Reach file:", ur10e_reach.__file__)
print("PGP file:", ur10e_pgp.__file__)
print("Registered environment:", gym.spec(ENVIRONMENT).entry_point)

#always keep device set to "cpu", "cuda" is not able to show simulations in real time/render_mode = "human"
env = gym.make(ENVIRONMENT, render_mode="human", max_episode_steps=MAX_EPISODE_STEPS, no_early_termination=NO_EARLY_TERMINATION)

print("Loaded environment module:", env.unwrapped.__class__.__module__)
#loading the algorithm model from checkpoint
if ALGORITHM == "SAC":
    model = SAC.load(MODEL_PATH, env=env, device="cpu")
    print("SAC model loaded from ", MODEL_PATH)
elif ALGORITHM == "PPO":
    model = PPO.load(MODEL_PATH, env=env, device="cpu")
    print("PPO model loaded from ", MODEL_PATH)
else:
    raise ValueError(f"Unknown algorithm: {ALGORITHM}")
observation, info = env.reset(seed=42) 

#when making the summary, which values do you want to track?
#when adding values, make shure they are also changed in the tester, otherwise they will not track correctly
component_names = [
    "distance_reward",
    "potential_reward",
    "orientation_reward",
    "success_ratio",
    "total_reward",
]

episode_rewards = []
episode_successes = []
success_ratio_history = []
success_ratio_episodes = []
episode_count = 0

summed_reward = 0.0
summed_components = {k: 0.0 for k in component_names}
all_episode_components = {k: [] for k in component_names}

for step in range(NUM_STEPS):
    #deterministic can be set to true when using SAC but when using PPO it must remain false otherwise the robot will act erratically
    #deterministic didn't cause any changes is SAC but major improvements in PPO if set to FALSE

    #got everything set up for replaying the results
    action, _states = model.predict(observation, deterministic=DETERMINISTIC)
    observation, reward, terminated, truncated, info = env.step(action)

    #tracking every reward and success
    summed_reward += reward
    components = info.get("reward_components", {})
    for k in ["distance_reward","potential_reward","orientation_reward","total_reward"]:
        summed_components[k] += components.get(k, 0.0)

    if terminated or truncated:
        episode_rewards.append(summed_reward)
        for k in component_names:
            all_episode_components[k].append(summed_components[k])

        episode_count += 1

        success = float(info.get("is_success", False))
        episode_successes.append(success)

        if episode_count % SUCCESS_LOG_EVERY == 0:
            success_rate = np.mean(episode_successes[-SUCCESS_LOG_EVERY:]) * 100.0
            success_ratio_history.append(success_rate)
            success_ratio_episodes.append(episode_count)
        elif success_ratio_history:
            success_rate = success_ratio_history[-1]
        else:
            success_rate = np.mean(episode_successes) * 100.0

        summed_components["success_ratio"] = success_rate
        ag_pos        = observation['achieved_goal'][:3]
        dg_pos        = observation['desired_goal'][:3]
        obj_pos       = observation['observation'][15:18]
        current_stage = int(observation['observation'][-1])
        dist_to_goal  = np.linalg.norm(ag_pos - dg_pos)
        dist_to_box   = np.linalg.norm(ag_pos - obj_pos)
        print(f"Episode {episode_count}: Reward={summed_reward:.2f}, "
              f"Success Rate={success_rate:.1f}%, Stage={current_stage}, "
              f"Dist to goal={dist_to_goal:.4f}m, "
              f"Dist to box={dist_to_box:.4f}m")

        summed_reward = 0.0
        summed_components = {k: 0.0 for k in component_names}
        observation, info = env.reset()

env.close()

#finished simulations, plotting the results
print("Finished all episodes, plotting results...")
episodes = list(range(1, episode_count + 1))
colors = plt.cm.tab10.colors

cols = 2
rows = (len(component_names) + 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
axes = axes.flatten()

for idx, k in enumerate(component_names):
    ax = axes[idx]
    if k == "success_ratio":
        ax.plot(success_ratio_episodes, success_ratio_history,
                marker='o', color=colors[idx % len(colors)], linewidth=2)
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(0,100)
    else:
        ax.plot(episodes, all_episode_components[k],
                marker='o', color=colors[idx % len(colors)], linewidth=2)
        ax.set_ylabel("Cumulative per episode")
    if k == "success_ratio":
        ax.set_xticks(success_ratio_episodes)
    else:
        ax.set_xticks(np.arange(
            SUCCESS_LOG_EVERY,
            episode_count + 1,
            SUCCESS_LOG_EVERY
        ))
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(k, fontsize=11)
    ax.set_xlabel("Episode")
    ax.grid(True)

for idx in range(len(component_names), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("Evaluation Metrics — UR10e", fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig("eval_rewards.png", dpi=150)
print("Graph saved to eval_rewards.png")
print("Close graph window to end")
plt.show()

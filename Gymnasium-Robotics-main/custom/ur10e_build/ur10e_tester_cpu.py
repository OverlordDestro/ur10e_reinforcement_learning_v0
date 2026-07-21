# Training script
# Creates the environment, initializes SAC/PPO,
# logs reward components and success ratio,
# periodically saves checkpoints,
# and produces training graphs.

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
import os

from stable_baselines3.common.callbacks import BaseCallback
import ur10e_pgp
import ur10e_reach

import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

ALGORITHM = "SAC"      # Change to "PPO" when needed
MAX_EPISODE_STEPS = 1000 #1000 simulation steps = 500 episode steps * 2 frame skip
EPISODES = 10000 #number of episodes to run the evaluation for, EPISODES * MAX_EPISODE_STEPS = NUM_STEPS
TOTAL_TIMESTEPS   = MAX_EPISODE_STEPS * EPISODES
#EPISODES and TOTAL_TIMESTEPS arent a good metric for how many episodes will actually run, as the simulator will always
#try to run TOTAL_TIMESTEPS, but if the simulation terminates/success before those MAX_EPISODE_STEPS are done
#those timesteps will be unused and instead be given to the next episode, simulator runs until it reaches TOTAL_TIMESTEPS

LOG_EVERY         = 50        # episodes until logging results
CHECKPOINT_EVERY  = 40        # episodes until checkpoint is saved
#checkpoints will always be created every CHECKPOINT_EVERY episodes, but a final checkpoint will also be made at the end
TRAIN_DEVICE      = "cpu" #set this to "cuda" or "cpu", cuda is far better and faster, but requires CUDA on your GPU
                            #only use cpu if your GPU does not cupport CUDA or want to watch the simulation in real time
                            #older or non NVIDIA hardware is not supported by CUDA
ENVIRONMENT = "UR10E-pgp-v0"     
MODEL_SAVE = "ur10e_pgp_SAC" #name of the model to save, will be saved in the current working directory
CHECKPOINT_SAVE = "ur10e_pgp_checkpoint" #name of the checkpoint, will be saved in the current working directory
N_ENVIRONMENTS = 1 #number of parallel environments to run, more environments = faster training, but requires more VRAM
CHECKPOINT_EVERY = 100 #number of episodes until a checkpoint is saved, checkpoints are saved in the current working directory
LOG_DIR = "./logs/" #for logging some info, can be removed if you want

print(f"Training device: {TRAIN_DEVICE}")
if TRAIN_DEVICE == "cuda":
    raise RuntimeError(f"Improper use of script, this script is ment for CPU but you are attempting to use it for GPU CUDA")
elif TRAIN_DEVICE == "cpu":
    print(f"Using CPU device")
else:
    raise RuntimeError(f"Improper use of script, unknown training device")

def make_env():
    return gym.make(ENVIRONMENT, render_mode="human", max_episode_steps=MAX_EPISODE_STEPS)

class RewardLoggerCallback(BaseCallback): 
    def __init__(self, log_every=LOG_EVERY, checkpoint_every=CHECKPOINT_EVERY, n_envs=1):
        super().__init__()
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.n_envs = n_envs
        self.ep_rew_means = []
        self.episode_rewards = []
        self.episode_successes = []
        self.current_ep_rewards = [0.0] * n_envs
        self.total_episode_count = 0

        # tracking components, change these if you want new components, make shure to also change it within the environment script
        self.component_names = [
            "distance_reward", "potential_reward", "orientation_reward",
            "total_reward"
        ]
        self.current_ep_components = [{k: 0.0 for k in self.component_names} for _ in range(n_envs)]
        self.ep_component_means = {k: [] for k in self.component_names}

    def _on_step(self):
        infos = self.locals.get("infos", [{}] * self.n_envs)

        for i in range(self.n_envs):
            self.current_ep_rewards[i] += self.locals["rewards"][i]

            # Accumulate reward components
            components = infos[i].get("reward_components", {})
            for k in self.component_names:
                self.current_ep_components[i][k] += components.get(k, 0.0)

            if self.locals["dones"][i]:
                self.total_episode_count += 1
                self.episode_rewards.append(self.current_ep_rewards[i])
                self.episode_successes.append(float(infos[i].get("is_success", False)))

                # Store per-episode component totals
                for k in self.component_names:
                    self.ep_component_means[k].append(self.current_ep_components[i][k])

                # Reset accumulators
                self.current_ep_rewards[i] = 0.0
                self.current_ep_components[i] = {k: 0.0 for k in self.component_names}

                if len(self.episode_rewards) % self.log_every == 0:
                    mean = np.mean(self.episode_rewards[-self.log_every:])
                    self.ep_rew_means.append(mean)
                    print(f"Episode {self.total_episode_count}: Mean reward (last {self.log_every}): {mean:.2f}")

                if self.total_episode_count % self.checkpoint_every == 0:
                    self.model.save(CHECKPOINT_SAVE)
                    print(f"💾 Checkpoint saved")

        return True

    def plot(self):
        colors = plt.cm.tab10.colors  # 10 distinct colors

        # ── Total reward plot ──────────────────────────────────────────────
        x = [(i + 1) * self.log_every for i in range(len(self.ep_rew_means))]
        plt.figure(figsize=(10, 5))
        plt.plot(x, self.ep_rew_means, color='steelblue')
        plt.xlabel("Episode")
        plt.ylabel(f"Mean reward (last {self.log_every} eps)")
        plt.title("Total Reward — SAC+HER on UR10e")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_reward_total.png", dpi=150)
        plt.show()

        # ── Per-component plots ────────────────────────────────────────────
        n = len(self.component_names) + 1  # reward components + success ratio
        cols = 2
        rows = (n + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
        axes = axes.flatten()

        episodes = list(range(1, len(next(iter(self.ep_component_means.values()))) + 1))

        for idx, k in enumerate(self.component_names):
            ax = axes[idx]
            color = colors[idx % len(colors)]
            data = np.array(self.ep_component_means[k])
            if len(data) >= self.log_every:
                smoothed = np.convolve(data, np.ones(self.log_every) / self.log_every, mode='valid')
                ax.plot(episodes[:len(smoothed)], smoothed, color=color, linewidth=2)
            else:
                ax.plot(episodes, data, color=color, linewidth=2)
            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_title(k)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Cumulative per episode")
            ax.grid(True)

        # Success ratio over the same rolling window used for reward smoothing.
        ax = axes[len(self.component_names)]
        successes = np.asarray(self.episode_successes, dtype=float)
        if len(successes) >= self.log_every:
            success_ratio = np.convolve(
                successes,
                np.ones(self.log_every) / self.log_every,
                mode="valid",
            )
            success_episodes = np.arange(self.log_every, len(successes) + 1)
        else:
            success_ratio = np.cumsum(successes) / np.arange(1, len(successes) + 1)
            success_episodes = np.arange(1, len(successes) + 1)
        ax.plot(success_episodes, success_ratio * 100.0, color=colors[4], linewidth=2)
        ax.set_title("success_ratio")
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Success rate (last {self.log_every} eps) [%]")
        ax.set_ylim(0, 100)
        ax.grid(True)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Reward Components and Success Ratio — SAC+HER on UR10e", fontsize=14)
        plt.tight_layout()
        plt.savefig("training_reward_components.png", dpi=150)
        plt.show()

#checking for log directory and creating it if it does not exist
os.makedirs(LOG_DIR, exist_ok=True) 

#creating environment
env = make_vec_env(make_env, n_envs=N_ENVIRONMENTS, vec_env_cls=DummyVecEnv, monitor_dir=LOG_DIR)

#settings for the algorithms,stable-baselines3 documentation has more info on specifics
if ALGORITHM == "SAC":
    model = SAC(
        "MultiInputPolicy", 
        env, 
        use_sde=True,  # set to false since in my testing it caused worse learning
        sde_sample_freq=4,  # sample a new noise value every 4 steps for smoother actions
        learning_starts=32000,  # 32 episodes for HER to collect enough transitions before learning starts
        buffer_size=500000,  # large buffer for better learning
        batch_size=512,  # large batch size for better learning
        gradient_steps=1,  # update the model every step
        tau=0.005,  # starndard tau
        device=TRAIN_DEVICE,    #keep on cuda for most cases, except for previously mentioned reasons, then set to cpu
        verbose=1,          #standard verbose
        learning_rate=1e-3, # standard learning rate
        gamma=0.99,          # 0.99 to hopefully improve learning, anything less was slowing learning
        ent_coef="auto" #keep on auto unless you have a specific use case between stages

    )
elif ALGORITHM == "PPO":#settings for PPO werent tested as much as SAC and havent been changed much from default
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=1e-3,
        n_steps=1024,                
        batch_size=64,               
        n_epochs=10,                 
        gamma=0.99,
        gae_lambda=0.95,             
        clip_range=0.2,              
        ent_coef=0.01,               
        vf_coef=0.5,                 
        max_grad_norm=0.5,           
        device=TRAIN_DEVICE,
        verbose=1,
    )
else:
    raise ValueError(f"Unsupported algorithm: {ALGORITHM}. Choose 'SAC' or 'PPO'.")

#starting simulations for training
callback = RewardLoggerCallback(log_every=N_ENVIRONMENTS, checkpoint_every=CHECKPOINT_EVERY, n_envs=N_ENVIRONMENTS)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# Save the trained model
model.save(MODEL_SAVE)
callback.plot()

print("Training complete, continue to evaluation")
print("close graphs to continue")
input("Press Enter to end")
print("Goodbye, world!")
env.close()

# UR10e SAC HER SB3 GYM MuJoCo Reinforcement Learning

Reinforcement learning for teaching a UR10e arm to learn and complete a **reach** task and a **pick and place** task.

This project uses SAC/PPO/TQC, curriculum learning, Gymnasium-Robotics, and Stable-Baselines3 within the MuJoCo simulator.

The subject is a **UR10e arm** with a **Robotiq 2F-85 gripper**.


---

## Collaboration

This project was made in collaboration with the **"Jožef Stefan" Institute**.

![Logo of the Institute of "Jožef Stefan".](Gymnasium-Robotics-main\gifs_images\IJS_logo_2.jpg)

---
## About
This project was created with the intent of assesing the appropriateness of MuJoCo to be used as a tool for further study of reinforcement learning. Within this project is supplied 2 examples of function: Reach, Pick&Place.
This file contains information on the project and setup, provided are also README_PGP,README_REACH and README_EXAMPLES for further details. Provided scripts work with the intent of mainly being used with SAC but there are options to use TQC and PPO. 

<table>
<tr>
<td align="center">
<img src="Gymnasium-Robotics-main/gifs_images/REACH_SAC_GIF.gif" 
</td>
SAC Reach
<td align="center">
<img src="Gymnasium-Robotics-main/gifs_images/PGP_SAC_GIF.gif" 
</td>
SAC P&P
</tr>
</table>
The repository also contains already trained models which can be inspected with the ur10e_loader.py. The models results are detailed in README_EXAMPLES to compare if they are working as intended for you.

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
| sb3-contrib | 2.9.0 |
---

## Project Files
This chapter defines important files used within the project
| File | Description |
| ---- | ----------- |
| `ur10e_reach.py` | Gymnasium MuJoCo environment for the reach task. Registers `UR10E-reach-v0`. |
| `ur10e_pgp.py` | Gymnasium MuJoCo environment for the pick-and-place task. Registers `UR10E-pgp-v0`. |
| `ur10e_tester.py` | Training script - CUDA, use to run training simulations |
| `ur10e_tester_cpu.py` | Training script — CPU, use to run training simulations |
| `ur10e_loader.py` | Evaluation script — loads a checkpoint and renders viewport to see results |
| `ur10e_gripper.xml` | MuJoCo XML model of the UR10e arm with Robotiq 2F-85 gripper |
| `ur10e_scene_mod.xml` | MuJoCo XML model of the UR10e arm, base for the ur10e_gripper.xml |
| `2f85.xml` | MuJoCo XML model of the 2f85 gripper from mujoco menagerie |
| `2f85_scene.xml` | MuJoCo XML model of the 2f85 gripper from mujoco menagerie, modified with ground |
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
## Setup

### 1. Clone the repository and install dependencies
python and git required, you can also copy ZIP instead of git clone
```bash
# clone repo
git clone https://github.com/OverlordDestro/ur10e_reinforcement_learning_v0.git
# go to folder
cd ur10e_reinforcement_learning_v0
#create virtual environment
python -m venv venv
#activate virtual environment
venv\Scripts\activate
or
source venv\Scripts\activate
#install dependencies
pip install numpy mujoco gymnasium gymnasium-robotics stable-baselines3 torch matplotlib sb3-contrib
#move to ur10e_build folder
cd Gymnasium-Robotics-main/custom/ur10e_build
#run tester
python ur10e_tester.py
```
or just copy ZIP 

### if you open in a new terminal later
```bash
cd /path/to/Gymnasium-Robotics-main
venv\Scripts\activate
or
source venv\Scripts\activate
cd Gymnasium-Robotics-main/custom/ur10e_build
```
to make it more convenient and easier to work on, I reccomend using VSC instead of in terminal
### 2. Train the model
when you have everything setup and are within the ur10e_build folder
you can start testing by running either ur10e_tester.py (reccomended) or ur10e_tester_cpu.py (very low spec)
```bash
python ur10e_tester.py
or
python ur10e_tester_cpu.py
```
change your prefered settings and which test to run (reach/pgp) within the ur10e_tester scripts
### 3. Load the checkpoint
when you have a model trained, you can set the name of the zip file to the trained model name within the loader

the loader will visually show you what the model learned and give you graphs on its performance in the end

make shure to check settings and change them to your test as always

Open `ur10e_loader.py` and ensure the correct checkpoint name is loaded:

```python
model = SAC.load("ur10e_checkpoint", env=env, device="cpu")
```

Then run:
```bash
python ur10e_loader.py
```
## Checkpoints
the tester scripts will periodically and at the end of simulation make checkpoints

these checkpoints will be saved in the root folder of the project 
(ur10e_reinforcement_learning_v0), the loader will also take from there

There are some examples of taught models in the root, which you can run in the loader, their performance graphs are also included

---
# RL algorithm settings
### SAC
SAC was chosen at the main algorithm for this research

| Hyperparameter | Value | Notes |
| -------------- | ----- | ----- |
| `learning_starts` | 32 000 | 32 episodes since doing 32 episodes at once |
| `buffer_size` | 500 000 | Large buffer for better learning |
| `batch_size` | 512 | Large batch for stable gradients |
| `gradient_steps` | 1 | One update per environment step |
| `tau` | 0.005 | Soft target network update coefficient |
| `learning_rate` | 1e-3 |  higher than default; found it to make learning faster 2x, but a small bit more unstable compared to 3e-4 |
| `gamma` | 0.99 | good value for future rewards |
| `ent_coef` | auto | Entropy coefficient — tuned automatically |
| `use_sde` | False | Found this to cause worse learning |
| `sde_sample_freq` | 4 | New noise sample every 4 steps |

### PPO

| Hyperparameter | Value |
| -------------- | ----- |
| `learning_rate` | 1e-3 |
| `n_steps` | 1 024 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `verbose` | 1 | 

### TQC
| Hyperparameter | Value |
| -------------- | ----- |
| `learning_rate` | 1e-3 |
| `learning_starts` | 32 000 |
| `buffer_size` | 500 000 |
| `batch_size` | 512 |
| `tau` | 0.005 |
| `gamma` | 0.99 |
| `train_freq` | 1 |
| `gradient_steps` | 1 | 
| `ent_coef` | auto |
| `use_sde` | False |
| `sde_sample_freq` | 4 |
| `top_quantiles_to_drop_per_net` | 2 |
| `verbose` | 1 | 
| `device` | TRAIN_DEVICE | 
---

## About PGP and Reach

See **[README_REACH.md](./README_REACH.md)** for Reach:
See **[README_PGP.md](./README_PGP.md)** for P&P:

## Example Results

See **[README_EXAMPLES.md](./README_EXAMPLES.md)** for examples:
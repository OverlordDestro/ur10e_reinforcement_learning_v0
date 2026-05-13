# ur10e_HER_SAC_SB3_GYM
ur10e is taught how to pick and place a box from a random position to a random target

all important files are in Gymnasium-Robotics-main/custom/ur10e_build

download dependencies from dependencies.txt

run ur10e_tester.py to start training, it will make checkpoints every 100 episodes, make a save at the end

run ur10e_loader.py to view results of checkpoint or save

stage 0 goes 20 cm above block, open gripper
stage 1 goes 5 cm above block, open gripper (intended to go directly ontop of block but for testing set it to 5cm above to avoid crashes to floor)
stage 2 closes gripper and lifts block above threshold

classic pick and place script for ur10e with 2f85 gripper but doesnt work beyond stage 0, even stage 0 is inconsistent and unprecise

uses cartesian system converted with a jacobian matrix from joints but has the same or worse results as when I used joints

uses mujoco with stable baselines 3, gymnasium robotics, SAC + HER (PPO got me worse results)

the main intend of the robot is (without locking any joint) to teach it to entend its arm without curling towards the box, have the TCP always looking downwards
and grab the box above the treshold, 20cm or something and hopefully eventually move it towards the target point

pip install gymnasium
pip install gymnasium_robotics
pip install numpy
pip install matplotlib
pip install stable_baselines3


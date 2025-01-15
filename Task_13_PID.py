from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env
import numpy as np

# Initialise the simulation environment
num_agents = 1
env = OT2Env(num_agents, render=True)
obs, info = env.reset()


### 
# 
#
#
# Do all the CV things so that you end up with a list of goal positions
#
#
#
###


# Load the trained agent
model = PPO.load("your trained RL model")

for goal_pos in goal_positions:
    # Set the goal position for the robot
    env.goal_position = root_pos
    # Run the control algorithm until the robot reaches the goal position
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info  = env.step(action)
        # calculate the distance between the pipette and the goal
        distance = obs[3:] - obs[:3] # goal position - pipette position
        # calculate the error between the pipette and the goal
        error = np.linalg.norm(distance)
        # Drop the inoculum if the robot is within the required error
        if error < 0.01: # 10mm is used as an example here it is too large for the real use case
            action = np.array([0, 0, 0, 1])
            obs, rewards, terminated, truncated, info  = env.step(action)
            break

        if terminated:
            obs, info = env.reset()




from stable_baselines3 import PPO
from ot2_gym_wrapper_V2 import OT2Env
import numpy as np

# Initialize the simulation environment
num_agents = 1
env = OT2Env(render=True)
obs, info = env.reset()

# Initialize the plate position and conversion factor
plate_position_robot = np.array([0.10775, 0.088 - 0.026, 0.057])  # Adjusted for the bug fix
plate_size_mm = 150
plate_size_pixels = 1000  # Replace with actual plate pixel dimensions
conversion_factor = plate_size_mm / plate_size_pixels

# Load the trained RL model
model = PPO.load("C:\\Users\\daanq\\Documents\\BUAS_Year_2B\\Block_B_Notes\\Tasks\\Task_11\\Final_model_it2.zip")

# Example computer vision output (pixel coordinates of root tips)
root_tips_pixel = [
    [200, 300],  # Replace with actual CV pipeline output
    [400, 600],
    [700, 800],
]

# Convert pixel coordinates to robot coordinates
root_tips_robot = []
for root_tip_pixel in root_tips_pixel:
    root_tip_mm = np.array(root_tip_pixel) * conversion_factor  # Convert to mm space
    root_tip_robot = plate_position_robot[:2] + root_tip_mm  # Add plate position offset
    root_tip_robot = np.append(root_tip_robot, plate_position_robot[2])  # Keep Z constant
    root_tips_robot.append(root_tip_robot)

# Iterate through each root tip position and inoculate
for goal_pos in root_tips_robot:
    env.goal_position = goal_pos  # Set goal position in the environment
    while True:
        # Use the RL model to predict the next action
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Calculate the distance to the goal
        distance = np.linalg.norm(obs[3:] - obs[:3])  # goal_position - pipette_position

        # If within the required error threshold, perform inoculation
        if distance < 0.001:  # Threshold set to 1 mm
            print(f"Inoculating at position: {goal_pos}")
            action = np.array([0, 0, 0, 1])  # Example inoculation action
            obs, rewards, terminated, truncated, info = env.step(action)
            break

        # Reset if terminated
        if terminated:
            obs, info = env.reset()



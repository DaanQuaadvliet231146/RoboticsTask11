import gymnasium as gym
import numpy as np
from ot2_gym_wrapper_V2 import OT2Env
from sim_class import Simulation
from PIL import Image  # For GIF creation

# Function to create a GIF from simulation frames
def create_simulation_gif(frames, filename="simulation.gif"):
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        filename, save_all=True, append_images=images[1:], duration=100, loop=0
    )
    print(f"GIF saved to {filename}")

# Number of episodes to simulate
num_episodes = 5

# Initialize your custom environment
env = OT2Env(render=True)
frames = []  # List to store frames for GIF creation

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0

    while not done:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Capture the simulation frame
        if hasattr(env.sim, "get_frame"):
            frame = env.sim.get_frame() 
            if frame is not None:
                frames.append(frame)

        # Print step details
        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")
        total_reward += reward
        step += 1

        # Check done condition
        done = terminated or truncated

        # Stop the episode at 1000 steps
        if step >= env.max_steps:
            print(f"Episode {episode + 1} reached maximum steps ({env.max_steps}).")
            break

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

# Save frames as a GIF if frames were captured
if frames:
    create_simulation_gif(frames, filename="simulation.gif")

env.close()

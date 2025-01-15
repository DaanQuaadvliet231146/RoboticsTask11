import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation  # Assuming sim_class provides the Simulation environment

import numpy as np

# X, Y bounding box
X_LOW, X_HIGH = -0.1872, 0.2531
Y_LOW, Y_HIGH = -0.1711, 0.2201

# Fixed Z (if you want to keep it constant)
Z_FIXED = 0.057


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.goal_position = np.zeros(3, dtype=np.float32)


        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action space (x, y, z movements with bounded velocities)
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)

        # Define observation space (pipette position + goal position)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Initialize step count
        self.steps = 0
        
    def get_plate_image(self):
    # Return the plate image path from the simulation
        return self.sim.plate_image_path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        self.sim.reset(num_agents=1)
        
        # Randomly pick X and Y from bounding box, keep Z fixed
        rand_x = np.random.uniform(X_LOW, X_HIGH)
        rand_y = np.random.uniform(Y_LOW, Y_HIGH)
        rand_z = Z_FIXED  # or any constant you like
        
        self.goal_position = np.array([rand_x, rand_y, rand_z], dtype=np.float32)
        
        pipette_position = np.array(
            self.sim.get_pipette_position(self.sim.robotIds[0]),
            dtype=np.float32
        )
        observation = np.concatenate((pipette_position, self.goal_position), axis=0)
        
        self.steps = 0
        info = {}
        return observation, info



    def step(self, action):
        # set the actions
        action = np.append(np.array(action, dtype=np.float32), 0)
        # Call the step function
        observation = self.sim.run([action])
        pipette_position = self.sim.get_pipette_position(self.sim.robotIds[0])
        # Process observation
        observation = np.array(pipette_position, dtype=np.float32)
        # Calculate the agent's reward
        distance = np.linalg.norm(np.array(pipette_position) - np.array(self.goal_position))
        reward = -distance
        
        # Check if the agent reaches within the threshold of the goal position
        if np.linalg.norm(pipette_position - self.goal_position) <= 0.001:
            terminated = True
        else:
            terminated = False

        # Check if episode should be truncated
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        observation = np.concatenate((pipette_position, self.goal_position), axis=0).astype(np.float32)
        info = {}

        # Update the amount of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()



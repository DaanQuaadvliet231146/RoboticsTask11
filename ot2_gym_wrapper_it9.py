# Import required packages
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

# Create the class
class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000, stagnation_threshold=5, stagnation_tolerance=0.0001):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        # Set variables for stagnation
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_tolerance = stagnation_tolerance

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Set the maximum values according to the working environment.
        self.x_min, self.x_max = -0.187, 0.2531
        self.y_min, self.y_max = -0.1705, 0.2195
        self.z_min, self.z_max = 0.1195, 0.2895
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_min, self.y_min, self.z_min, -self.x_max, -self.y_max, -self.z_max], dtype=np.float32),
            high=np.array([self.x_max, self.y_max, self.z_max, self.x_max, self.y_max, self.z_max], dtype=np.float32),
            dtype=np.float32
        )
        # Keep track of the step amount
        self.steps = 0
        
    def get_plate_image(self):
        return self.sim.get_plate_image()
    
    def reset(self, seed=None):
        # Set a seed if it was not set yet
        if seed is not None:
            np.random.seed(seed)

        # Randomise the goal position
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        z = np.random.uniform(self.z_min, self.z_max)
        # Set a random goal position
        self.goal_position = np.array([x, y, z])
        # Call reset function
        observation = self.sim.reset(num_agents=1)
        # Set the observation.
        observation = np.concatenate(
            (
                self.sim.get_pipette_position(self.sim.robotIds[0]), 
                self.goal_position
            ), axis=0
        ).astype(np.float32) 

        # Reset the number of steps
        self.steps = 0
        self.previous_positions = []

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
        # Ensure the right length for the previous positions variable
        if len(self.previous_positions) >= self.stagnation_threshold:
            self.previous_positions.pop(0)
        self.previous_positions.append(pipette_position)
        # If the positions get above the threshold for stagnation.
        if len(self.previous_positions) == self.stagnation_threshold:
            position_deltas = np.linalg.norm(np.diff(self.previous_positions, axis=0), axis=1)
            # Check for stagnation.
            if np.all(position_deltas <= self.stagnation_tolerance):
                # Give a penalty for stagnation.
                reward-=10
        
        # Check if the agent reaches within the threshold of the goal position, ignore z-axis due to irrelevancy.
        if np.linalg.norm(pipette_position[:2] - self.goal_position[:2]) <= 0.001:
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
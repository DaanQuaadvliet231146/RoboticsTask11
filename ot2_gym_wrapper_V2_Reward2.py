import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation  # Assuming sim_class provides the Simulation environment

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

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

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset simulation environment
        initial_obs = self.sim.reset(num_agents=1)

        # Randomly set a new goal position
        self.goal_position = np.array([
            np.random.uniform(0.2531, -0.1872),
            np.random.uniform(0.2201, -0.1711),
            np.random.uniform(0.2896, 0.1691)         
            ])

        # Call reset function
        observation = self.sim.reset(num_agents=1)
        # Set the observation.
        observation = np.concatenate(
            (
                self.sim.get_pipette_position(self.sim.robotIds[0]), 
                self.goal_position
            ), axis=0
        ).astype(np.float32) 

        # Reset step counter
        self.steps = 0

        # Return observation and info
        return observation, {}



    def step(self, action):
    # Set the actions
        action = np.append(np.array(action, dtype=np.float32), 0)
        # Call the step function
        observation = self.sim.run([action])
        pipette_position = self.sim.get_pipette_position(self.sim.robotIds[0])
        # Process observation
        observation = np.array(pipette_position, dtype=np.float32)
        
        # Calculate the agent's reward (Exponential Decay)
        distance = np.linalg.norm(np.array(pipette_position) - np.array(self.goal_position))
        reward = -np.exp(distance * 10)  # Adjust the factor (10) for steepness of penalty
        
        # Check if the agent reaches within the threshold of the goal position
        terminated = np.linalg.norm(pipette_position - self.goal_position) <= 0.001

        # Check if episode should be truncated
        truncated = self.steps >= self.max_steps

        # Concatenate observation
        observation = np.concatenate((pipette_position, self.goal_position), axis=0).astype(np.float32)
        info = {}

        # Update the amount of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()



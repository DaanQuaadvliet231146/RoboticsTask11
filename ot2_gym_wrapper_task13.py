import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation  # Assuming sim_class provides the Simulation environment

GOAL_LOW = np.array([-0.1872, -0.1711, 0.1691], dtype=np.float32)
GOAL_HIGH = np.array([ 0.2531,  0.2201, 0.2896], dtype=np.float32)


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
    
    def some_method(self):
        x = np.random.uniform(-0.1872, 0.2531)
        y = np.random.uniform(-0.1711, 0.2201)
        z = 0.057
        self.goal_position = [x, y, z]

    def reset(self, goal_position=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset simulation environment
        initial_obs = self.sim.reset(num_agents=1)

            
        # Decide how to set self.goal_position
        if goal_position is not None:
            # Clip user-provided goal_position to bounding box
            clipped_goal = np.clip(goal_position, GOAL_LOW, GOAL_HIGH)
            self.goal_position = clipped_goal.astype(np.float32)
        else:
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
        # set the actions
        action = np.append(np.array(action, dtype=np.float32), 0)
        # Call the step function
        observation = self.sim.run([action])
        pipette_position = self.sim.get_pipette_position(self.sim.robotIds[0])
        pipette_position = np.array(pipette_position, dtype=np.float32)

        # Process observation
        observation = np.array(pipette_position, dtype=np.float32)
        # Calculate the agent's reward
        distance = np.linalg.norm(pipette_position - self.goal_position)
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

    def get_plate_image(self):
    # Return the plate image path from the simulation
        return self.sim.plate_image_path




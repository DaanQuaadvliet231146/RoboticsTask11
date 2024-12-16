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
            ])  # Example bounds

        # Extract pipette position and create the observation array
        robot_key = next(iter(initial_obs))  # Dynamically get the first key
        pipette_position = np.array(initial_obs[robot_key]['pipette_position'], dtype=np.float32)
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # Reset step counter
        self.steps = 0

        # Return observation and info
        return observation, {}



    def step(self, action):
        # Append a zero drop action (assuming the simulator expects 4 values)
        extended_action = np.append(action, [0])

        # Execute action in simulation
        sim_obs = self.sim.run([extended_action])
        print("sim_obs:", sim_obs)  # Debugging: Print the simulation output

        # Access the robot data directly
        robot_key = next(iter(sim_obs))  # Dynamically get the first key (e.g., 'robotId_1')
        robot_data = sim_obs[robot_key]
        pipette_position = np.array(robot_data['pipette_position'], dtype=np.float32)

        # Create observation
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)

        # Reward is negative distance
        reward = -distance_to_goal

        # Check termination condition
        terminated = bool(distance_to_goal < 0.05)  # Explicitly cast to boolean

        truncated = self.steps >= self.max_steps

        # Increment step count
        self.steps += 1

        return observation, reward, terminated, truncated, {}




    def render(self, mode='human'):
        if self.render:
            self.sim.render()

    def close(self):
        self.sim.close()



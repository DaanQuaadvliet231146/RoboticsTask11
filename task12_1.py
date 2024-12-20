from simple_pid import PID
import numpy as np
import gymnasium as gym
from sim_class import Simulation  # Your simulation environment

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Initialize simple-pid controllers
        self.pid_x = PID(1.0, 0.01, 0.1, setpoint=0.0)
        self.pid_y = PID(1.0, 0.01, 0.1, setpoint=0.0)
        self.pid_z = PID(1.0, 0.01, 0.1, setpoint=0.0)

        # Set output limits to prevent overly large velocity commands
        for pid in [self.pid_x, self.pid_y, self.pid_z]:
            pid.output_limits = (-1.0, 1.0)  # Constrain velocities between -1 and 1

        # Simulation setup
        self.sim = Simulation(num_agents=1, render=self.render)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation and set a random goal position
        initial_obs = self.sim.reset(num_agents=1)
        self.goal_position = np.random.uniform([0.2, 0.2, 0.2], [0.5, 0.5, 0.5])
        robot_key = next(iter(initial_obs))
        pipette_position = np.array(initial_obs[robot_key]['pipette_position'], dtype=np.float32)

        # Update PID setpoints
        self.pid_x.setpoint = self.goal_position[0]
        self.pid_y.setpoint = self.goal_position[1]
        self.pid_z.setpoint = self.goal_position[2]

        # Initialize state tracking
        self.previous_distance = np.linalg.norm(pipette_position - self.goal_position)
        self.steps = 0

        return np.concatenate((pipette_position, self.goal_position)), {}

    def step(self, _):
        # Get the current pipette position
        sim_obs = self.sim.get_states()
        robot_key = next(iter(sim_obs))
        pipette_position = np.array(sim_obs[robot_key]['pipette_position'], dtype=np.float32)

        # Compute PID outputs (velocities for each axis)
        vel_x = self.pid_x(pipette_position[0])
        vel_y = self.pid_y(pipette_position[1])
        vel_z = self.pid_z(pipette_position[2])

        # Apply the velocities in the simulation
        self.sim.run([[vel_x, vel_y, vel_z, 0]], num_steps=1)

        # Compute rewards
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        reward = self.previous_distance - distance_to_goal  # Reward for reducing distance
        self.previous_distance = distance_to_goal

        # Check termination and truncation
        terminated = distance_to_goal < 0.001  # Terminate if within 1 mm of the goal
        truncated = self.steps >= self.max_steps
        self.steps += 1

        # Return updated observation
        observation = np.concatenate((pipette_position, self.goal_position))
        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if self.render:
            self.sim.render()

    def close(self):
        self.sim.close()
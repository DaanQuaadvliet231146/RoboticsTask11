import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation  # Where your bullet/pipeline simulation is

class OT2Env(gym.Env):
    """
    A custom Gym environment that steps the OT-2 robot in (x, y, z) velocities
    and observes [pipette_position, goal_position].
    """

    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render_mode = render
        self.max_steps = max_steps

        # Create the underlying simulation environment
        # (Assumes sim_class.Simulation has a constructor that takes num_agents=1, render=bool)
        self.sim = Simulation(num_agents=1, render=self.render_mode)

        # Action space: [v_x, v_y, v_z], velocities between -1 and +1 m/s (adjust as you like)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]), 
            high=np.array([ 1,  1,  1]), 
            dtype=np.float32
        )

        # Observation space: 6D = [pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z]
        # You could also limit it if you know the bounds, but we'll just use +/- inf for simplicity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Store the current step count
        self.num_steps = 0

        # Will store the [x, y, z] of the goal
        self.goal_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_plate_image(self):
    # Return the plate image path from the simulation
        return self.sim.plate_image_path
    
    
    def reset(self, seed=None, options=None):
        """
        Reset the simulation, pick a default or random goal (if desired), 
        and return the initial observation.
        """
        super().reset(seed=seed)

        # Reset the bullet simulation
        self.sim.reset(num_agents=1)
        self.num_steps = 0

        # For PID usage, we might externally set self.goal_position
        # or randomize it if we prefer. For now let's keep it at [0,0,0].
        # (You can override this logic in your main code.)

        # Build the new observation
        pipette_pos = self.sim.get_pipette_position(self.sim.robotIds[0])  # shape (3,)
        obs = np.concatenate([pipette_pos, self.goal_position], axis=0).astype(np.float32)
        return obs, {}

    def step(self, action):
        """
        Step the simulation with the given velocity in x/y/z.
        We also compute a simple 'reward' as negative distance to goal 
        (though for PID, you may not actually use reward).
        """
        self.num_steps += 1

        # Make sure action is shape (3,) => [v_x, v_y, v_z]
        if len(action) == 3:
            # The environment might expect a 4th index if there's a gripper or 
            # 'drop inoculum' dimension. For PID movement only, we can pass 0 as the 4th:
            action = np.append(action, 0.0)  # No "gripper" action

        # Actually run the simulation
        new_obs = self.sim.run([action])

        # Get the new pipette position as a NumPy array
        pipette_pos = np.array(self.sim.get_pipette_position(self.sim.robotIds[0]), dtype=np.float32)

        # Convert the goal position to a NumPy array if it isn't already
        goal_position = np.array(self.goal_position, dtype=np.float32)

        # Compute the Euclidean distance between the pipette position and the goal
        dist = np.linalg.norm(pipette_pos - goal_position)
        reward = -dist  # A simple negative distance reward

        # Check termination conditions
        terminated = False
        # If we are within 1 mm => done
        if dist <= 0.001:
            terminated = True

        truncated = False
        # If we exceed the max steps => truncated
        if self.num_steps >= self.max_steps:
            truncated = True

        # Build the new observation = [pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z]
        obs = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        # Not strictly needed for the bullet-based OT2 simulator.
        pass

    def close(self):
        self.sim.close()

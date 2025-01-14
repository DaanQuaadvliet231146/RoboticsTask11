from simple_pid import PID
import numpy as np
from ot2_gym_wrapper import OT2Env
import random


class PIDRandomPositionSimulation:
    def __init__(self):
        # Initialize the OT2 Gym Environment
        self.env = OT2Env(render=True)
        self.dt = 0.1  # Reduced time step for finer control

        # Initialize PID controllers for each axis with updated gains
        self.pid_x = PID(40.0, 0.1, 1.2, sample_time=self.dt)
        self.pid_y = PID(40.0, 0.1, 1.2, sample_time=self.dt)
        self.pid_z = PID(40.0, 0.1, 1.2, sample_time=self.dt)

        # Set PID output limits (velocity bounds)
        for pid in [self.pid_x, self.pid_y, self.pid_z]:
            pid.output_limits = (-4.0, 4.0)

        # Define bounds of the working envelope
        self.bounds = {
            "x": [0.2531, -0.1872],
            "y": [0.2201, -0.1711],
            "z": [0.2896, 0.1691],
        }

    def move_to_random_position(self):
        # Generate a random target position within the bounds
        target = [
            random.uniform(self.bounds["x"][1], self.bounds["x"][0]),
            random.uniform(self.bounds["y"][1], self.bounds["y"][0]),
            random.uniform(self.bounds["z"][1], self.bounds["z"][0]),
        ]

        print(f"Moving to Random Position: {target}")

        # Set PID setpoints for the target position
        self.pid_x.setpoint = target[0]
        self.pid_y.setpoint = target[1]
        self.pid_z.setpoint = target[2]

        # Reset environment and get initial position
        observation, _ = self.env.reset()
        pipette_position = observation[:3]

        for step in range(1000):  # Limit steps to 1000
            # Compute PID-controlled velocities
            vel_x = self.pid_x(pipette_position[0])
            vel_y = self.pid_y(pipette_position[1])
            vel_z = self.pid_z(pipette_position[2])

            # Apply reduced velocity limits near the target
            distance_to_target = np.linalg.norm(pipette_position - target)
            if distance_to_target < 0.01:
                self.pid_x.output_limits = (-1.0, 1.0)
                self.pid_y.output_limits = (-1.0, 1.0)
                self.pid_z.output_limits = (-1.0, 1.0)

            # Apply velocities to the simulation
            action = [vel_x, vel_y, vel_z]
            observation, _, _, _, _ = self.env.step(action)

            # Update pipette position
            pipette_position = observation[:3]

            # Log progress
            if step % 50 == 0 or distance_to_target < 0.001:
                print(f"Step {step}: Distance to target = {distance_to_target:.4f} m")

            # Early stop if target is reached
            if distance_to_target < 0.001:  # 1 mm accuracy
                print(f"Target reached in {step+1} steps.")
                return step + 1, target, distance_to_target

        print(f"Failed to reach target within 1000 steps. Final distance: {distance_to_target:.4f} m")
        return 1000, target, distance_to_target

    def simulate(self, num_positions=5):
        results = []
        for i in range(num_positions):
            steps, target, final_distance = self.move_to_random_position()
            status = "Success" if final_distance < 0.001 else "Failure"
            print(f"Position {i+1} | Steps: {steps} | Target: {target} | Status: {status} | Final Distance: {final_distance:.4f} m")
            results.append((target, status, final_distance))

        return results

    def close(self):
        self.env.close()


if __name__ == "__main__":
    simulation = PIDRandomPositionSimulation()
    results = simulation.simulate(num_positions=5)
    simulation.close()

    print("\nSimulation Results:")
    for i, (target, status, final_distance) in enumerate(results):
        print(f"Position {i+1}: Target {target} | Status: {status} | Final Distance: {final_distance:.4f} m")

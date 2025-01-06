import os
import numpy as np
from stable_baselines3 import PPO
from ot2_gym_wrapper_Task12 import OT2Env

if __name__ == "__main__":
    # Path to the pretrained PPO model
    model_path = "C:\\Users\\daanq\\Documents\\BUAS_Year_2B\\Block_B_Notes\\Tasks\\Task_11\\model_16384.zip"

    # Load the OT2 environment
    env = OT2Env(render=True)

    # Load the pretrained PPO model
    model = PPO.load(model_path)

    # Run simulation
    num_episodes = 5
    results = []

    for episode in range(num_episodes):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0

        # Log target position
        target_position = observation[3:]
        print(f"Moving to Random Position: {target_position.tolist()}")
        while not (terminated or truncated):
            # Predict the action using the loaded PPO model
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            steps += 1
            print(f"Action at step {steps}: {action}")


        final_position = observation[:3]
        distance_to_target = np.linalg.norm(final_position - target_position)
        status = "Success" if distance_to_target < 0.001 else "Failure"

        print(f"Target reached in {steps} steps.")
        print(f"Final Position: {final_position.tolist()}, Distance to Target: {distance_to_target:.4f} m")
        print(f"Position {episode + 1} | Steps: {steps} | Target: {target_position.tolist()} | Status: {status} | Final Distance: {distance_to_target:.4f} m")

        results.append((steps, target_position.tolist(), status, distance_to_target))

    env.close()

    print("\nSimulation Results:")
    for i, (steps, target_position, status, final_distance) in enumerate(results):
        print(f"Position {i + 1}: Target {target_position} | Status: {status} | Final Distance: {final_distance:.4f} m")

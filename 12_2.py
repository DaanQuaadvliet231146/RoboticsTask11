from task12_1 import OT2Env
import matplotlib.pyplot as plt

# Initialize the environment
env = OT2Env(render=False)
observation, _ = env.reset()

# Track data for plotting
positions = []
rewards = []

for _ in range(1000):  # Run for up to 1000 steps
    observation, reward, terminated, truncated, _ = env.step(None)
    positions.append(observation[:3])  # Record pipette position
    rewards.append(reward)

    if terminated or truncated:
        break

env.close()

# Plot the results
positions = np.array(positions)
plt.figure(figsize=(12, 6))
plt.plot(positions[:, 0], label="X-axis")
plt.plot(positions[:, 1], label="Y-axis")
plt.plot(positions[:, 2], label="Z-axis")
plt.axhline(env.goal_position[0], color="r", linestyle="--", label="Goal X")
plt.axhline(env.goal_position[1], color="g", linestyle="--", label="Goal Y")
plt.axhline(env.goal_position[2], color="b", linestyle="--", label="Goal Z")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.legend()
plt.title("Pipette Position vs. Goal Position")
plt.show()

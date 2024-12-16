import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env

# Step 1: Verify Environment
env = OT2Env(render=False)
check_env(env)

# Step 2: Test Random Actions
num_episodes = 5
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0

    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Obs={obs}, Reward={reward}, Terminated={terminated}, Truncated={truncated}")

        if terminated or truncated:
            print(f"Episode {episode + 1} finished after {step + 1} steps.")
            break
        step += 1
    
env.close()

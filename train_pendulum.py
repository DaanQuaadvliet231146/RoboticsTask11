from stable_baselines3 import PPO
import argparse
import sys
import gymnasium as gym
import time
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import os
os.environ['WANDB_API_KEY'] = 'a17490878d1f01a688053dab318a1001adac6f82'
os.environ["WANDB_SYMLINK"] = "false"

env = gym.make('Pendulum-v1',g=9.81)

# initialize wandb project
run = wandb.init(project="sb3_pendulum_demo",sync_tensorboard=True)

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)

args, unknown = parser.parse_known_args()

# add tensorboard logging to the model
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}",)

# create wandb callback
wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

# add wandb callback to the model training
model.learn(total_timesteps=10000, callback=wandb_callback, progress_bar=True, tb_log_name=f"runs/{run.id}")
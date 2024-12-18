from stable_baselines3 import PPO
import argparse
import sys
import gymnasium as gym
import time
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import os
from clearml import Task
from typing_extensions import TypeIs
import tensorflow
from ot2_gym_wrapper import OT2Env


# Use the appropriate project name and task name (if you are in the first group in Dean's mentor group, use the project name 'Mentor Group D/Group 1')
# It can also be helpful to include the hyperparameters in the task name
task = Task.init(project_name='Mentor Group K/Group 1', task_name='Experiment1_DaanQuaadvliet')
#copy these lines exactly as they are
#setting the base docker image
wrapped_env = OT2Env()

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

os.environ['WANDB_API_KEY'] = 'a17490878d1f01a688053dab318a1001adac6f82'
os.environ["WANDB_SYMLINK"] = "false"

env = gym.make('Pendulum-v1',g=9.81)

# initialize wandb project
run = wandb.init(project="sb3_pendulum_demo",sync_tensorboard=True)

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_steps", type=int, default=550000)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.98)
parser.add_argument("--gae_lambda", type=float, default=0.9)
parser.add_argument("--value_coefficient", type=float, default=0.5)
parser.add_argument("--target_kl", type=float, default=0.02)

args, unknown = parser.parse_known_args()

# Define the PPO model
model = PPO(
    "MlpPolicy", wrapped_env, verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    vf_coef=args.value_coefficient,
    target_kl=args.target_kl,
    tensorboard_log=f"runs/{run.id}",
)


# create wandb callback
wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

# add wandb callback to the model training
model.learn(total_timesteps=5000000, callback=wandb_callback, progress_bar=True, tb_log_name=f"runs/{run.id}")
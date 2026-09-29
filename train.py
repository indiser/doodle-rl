from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from doodle_env import DoodleEnv

env = DoodleEnv()
check_env(env)  # crash here = bug in env, fix before training

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("doodle_ppo")
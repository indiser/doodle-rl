from stable_baselines3 import PPO
from doodle_env import DoodleEnv

env = DoodleEnv(render_mode="human")
model = PPO.load("doodle_ppo")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    if terminated or truncated:
        print("score", info["score"])
        obs, _ = env.reset()
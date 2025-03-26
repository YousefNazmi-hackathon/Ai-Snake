import gym
from stable_baselines3 import PPO
from snake_env import SnakeEnv

# Load the trained model
model = PPO.load("ppo_snake")

# Create the environment
env = SnakeEnv()

# Evaluate the agent
episodes = 10
for episode in range(episodes):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs)  # Get action from the trained model
        obs, reward, done, info = env.step(action)
        score += reward
        env.render()  # Render the environment

    print(f"Episode {episode + 1}: Score = {score}")

env.close()

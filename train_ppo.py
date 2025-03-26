import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from snake_env import SnakeEnv

# Create the environment
env = SnakeEnv()

# Wrap the environment for vectorized training
vec_env = make_vec_env(lambda: env, n_envs=4)  # Use 4 parallel environments for faster training

# Define the PPO model
model = PPO(
    "MlpPolicy",  # Use a multi-layer perceptron policy
    vec_env,
    verbose=1,
    learning_rate=0.0001,  # Lower learning rate for stable training
    n_steps=1024,  # Reduce steps for faster updates
    batch_size=128,  # Larger batch size for better gradient estimates
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_snake_tensorboard/"
)

# Train the model
model.learn(total_timesteps=1_000_000)  # Train for 1 million timesteps

# Save the trained model
model.save("ppo_snake")

# Close the environment
env.close()

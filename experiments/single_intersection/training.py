import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from sumo_traffic_env import SumoTrafficEnv  # Import your custom gym.Env wrapper

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit('Declare environment variable "SUMO_HOME"')


# Create the environment
env = SumoTrafficEnv()

# Optional: Check environment sanity
check_env(env)

# Wrap in a vectorized environment
vec_env = DummyVecEnv([lambda: env])

# Define PPO agent
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    clip_range=0.2,
    n_epochs=10,
    tensorboard_log="./ppo_sumo_tensorboard/"
)

# Train agent
model.learn(total_timesteps=100_000)  # Adjust depending on resources and convergence

# Save model
model.save("ppo_traffic_model_single_intersection")
print("✅ Model trained and saved as 'ppo_traffic_model_single_intersection.zip'")

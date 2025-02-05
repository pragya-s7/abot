import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class ModularRobotPyBulletEnv(gym.Env):
    def __init__(self, gui=True):
        super(ModularRobotPyBulletEnv, self).__init__()
        
        # Enable GUI if specified, or switch to direct mode
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Ensure PyBullet has access to necessary resources
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load environment models and robot
        self.robot = p.loadURDF("r2d2.urdf", useFixedBase=True)
        p.setGravity(0, 0, -9.8)
        
        # Set observation and action spaces (adjust as needed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        # Reset the simulation environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.robot = p.loadURDF("r2d2.urdf", useFixedBase=True)
        return np.zeros(6, dtype=np.float32)
    
    def step(self, action):
        # Apply action to the robot (replace this with meaningful control logic)
        p.stepSimulation()
        time.sleep(0.01)  # Slow down the simulation if needed
        
        # Mock observation, reward, done condition
        observation = np.zeros(6, dtype=np.float32)
        reward = -np.linalg.norm(action)  # Example negative reward
        done = False  # Define terminal condition based on task
        return observation, reward, done, {}

    def render(self, mode="human"):
        # Render the environment using PyBullet's GUI
        pass

    def close(self):
        p.disconnect()


# Main script to train the model using Stable-Baselines3 PPO
if __name__ == "__main__":
    # Initialize environment with GUI enabled
    env = ModularRobotPyBulletEnv(gui=True)

    # Wrap the environment in a vectorized environment for Stable Baselines3
    env = DummyVecEnv([lambda: env])

    # Initialize and train PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2048)

    # Test the model in the environment
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        # Display step rewards
        print(f"Step Reward: {rewards}")

    env.close()

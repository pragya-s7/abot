import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------------------
# 1) Define Robot Modules
# ---------------------------
class ModuleType:
    CORE = 0
    LOCOMOTION = 1
    MANIPULATOR = 2
    SENSOR = 3

    @staticmethod
    def name(module_id):
        return {
            ModuleType.CORE: "Core",
            ModuleType.LOCOMOTION: "Locomotion",
            ModuleType.MANIPULATOR: "Manipulator",
            ModuleType.SENSOR: "Sensor"
        }.get(module_id, "Unknown")


# ---------------------------
# 2) Define Tasks
# ---------------------------
# We'll define a simple "move to goal" task. 
# You can expand this to picking/placing or obstacle navigation.

class Task:
    def __init__(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y

    def get_goal_position(self):
        return np.array([self.goal_x, self.goal_y])


# ---------------------------
# 3) Create the RL Environment
# ---------------------------
class ModularRobotEnv(gym.Env):
    """
    A simplified environment to demonstrate:
    - Adding/removing modules
    - Moving the robot toward a goal
    - Rewarding minimal module usage + quick goal reach
    """

    def __init__(self, max_modules=5, world_size=10.0, max_steps=50):
        super(ModularRobotEnv, self).__init__()
        
        # Environment parameters
        self.max_modules = max_modules
        self.world_size = world_size
        self.max_steps = max_steps

        # Define the observation space:
        # [robot_x, robot_y, goal_x, goal_y, #core, #locomotion, #manipulator, #sensor]
        # We'll keep all values in a certain range to be consistent.
        high = np.array([
            world_size, world_size,  # robot_x, robot_y
            world_size, world_size,  # goal_x,  goal_y
            max_modules, max_modules, max_modules, max_modules  # module counts
        ], dtype=np.float32)
        low = -high

        self.observation_space = spaces.Box(low=low, high=high, shape=(8,), dtype=np.float32)

        # Define the action space:
        # 0 -> add locomotion module
        # 1 -> remove locomotion module
        # 2 -> add manipulator module
        # 3 -> remove manipulator module
        # 4 -> add sensor module
        # 5 -> remove sensor module
        # 6 -> move left
        # 7 -> move right
        # 8 -> move up
        # 9 -> move down
        self.action_space = spaces.Discrete(10)

        # Internal state
        self.reset()

    def reset(self):
        # Reset the episode
        self.step_count = 0

        # Randomize robot's start position
        self.robot_pos = np.random.uniform(-self.world_size / 2, self.world_size / 2, size=2)

        # Create a random task
        goal_x = np.random.uniform(-self.world_size / 2, self.world_size / 2)
        goal_y = np.random.uniform(-self.world_size / 2, self.world_size / 2)
        self.task = Task(goal_x, goal_y)

        # Start with 1 core module by default
        # (Assume exactly 1 core is always needed, so we'll keep that count = 1 and not let user remove it.)
        self.num_core = 1
        self.num_locomotion = 0
        self.num_manipulator = 0
        self.num_sensor = 0

        return self._get_observation()

    def step(self, action):
        self.step_count += 1

        # Process the action
        if action == 0:  # add locomotion
            if self._can_add_module():
                self.num_locomotion += 1
        elif action == 1:  # remove locomotion
            if self.num_locomotion > 0:
                self.num_locomotion -= 1
        elif action == 2:  # add manipulator
            if self._can_add_module():
                self.num_manipulator += 1
        elif action == 3:  # remove manipulator
            if self.num_manipulator > 0:
                self.num_manipulator -= 1
        elif action == 4:  # add sensor
            if self._can_add_module():
                self.num_sensor += 1
        elif action == 5:  # remove sensor
            if self.num_sensor > 0:
                self.num_sensor -= 1
        elif action == 6:  # move left
            self._move_robot(dx=-1, dy=0)
        elif action == 7:  # move right
            self._move_robot(dx=1, dy=0)
        elif action == 8:  # move up
            self._move_robot(dx=0, dy=1)
        elif action == 9:  # move down
            self._move_robot(dx=0, dy=-1)

        # Calculate reward
        distance_to_goal = self._distance_to_goal()
        done = False

        # Reward for being closer to goal
        # We'll use negative distance as part of the reward to encourage minimizing distance
        reward = -distance_to_goal

        # Small penalty per additional module (beyond the required 1 core) to encourage minimal configurations
        total_modules = self.num_core + self.num_locomotion + self.num_manipulator + self.num_sensor
        reward -= 0.1 * (total_modules - 1)

        # Check if goal is reached
        if distance_to_goal < 0.5:  # within some threshold
            reward += 10.0
            done = True

        # Check if max steps exceeded
        if self.step_count >= self.max_steps:
            done = True

        observation = self._get_observation()
        info = {}
        return observation, reward, done, info

    def _get_observation(self):
        return np.array([
            self.robot_pos[0],
            self.robot_pos[1],
            self.task.goal_x,
            self.task.goal_y,
            float(self.num_core),
            float(self.num_locomotion),
            float(self.num_manipulator),
            float(self.num_sensor)
        ], dtype=np.float32)

    def _distance_to_goal(self):
        goal_pos = self.task.get_goal_position()
        return np.linalg.norm(self.robot_pos - goal_pos)

    def _move_robot(self, dx, dy):
        # Movement speed can depend on locomotion modules, for instance.
        base_speed = 0.1
        # For example: more locomotion modules -> faster movement
        speed_factor = 1.0 + 0.1 * self.num_locomotion
        self.robot_pos[0] += dx * base_speed * speed_factor
        self.robot_pos[1] += dy * base_speed * speed_factor

        # Clip position within world bounds
        self.robot_pos = np.clip(self.robot_pos, -self.world_size / 2, self.world_size / 2)

    def _can_add_module(self):
        total_modules = self.num_core + self.num_locomotion + self.num_manipulator + self.num_sensor
        return total_modules < self.max_modules


# ---------------------------
# 4) Training the RL Agent
# ---------------------------
def train_agent(total_timesteps=5000):
    # Create environment and wrap it for vectorized handling
    env = ModularRobotEnv()
    vec_env = DummyVecEnv([lambda: env])

    # Initialize PPO (you can also try DQN, A2C, etc.)
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # Train the model
    model.learn(total_timesteps=total_timesteps)
    return model

def evaluate_agent(model, num_episodes=5):
    # Evaluate the trained agent for a few episodes
    env = ModularRobotEnv()
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        print(f"Episode {ep + 1} reward: {episode_reward:.2f}")


if __name__ == "__main__":
    # Example usage:
    print("Training the agent...")
    trained_model = train_agent(total_timesteps=10000)

    print("\nEvaluating the agent...")
    evaluate_agent(trained_model, num_episodes=5)

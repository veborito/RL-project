import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from game_env.snake_game import SnakeGame

class SnakeEnv(gym.Env):
    """Gymnasium environment wrapper for Snake game"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, width=20, height=15):
        super().__init__()

        self.width = width
        self.height = height
        self.render_mode = render_mode

        # Initialize the game
        self.game = SnakeGame(width=width, height=height)

        # Define action and observation space
        # Actions: 4 directions (up, right, down, left)
        self.action_space = spaces.Discrete(4)

        # Observation space: 11 boolean/binary features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float32
        )

        # Initialize pygame if render mode is human
        if self.render_mode == "human":
            pygame.init()

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        observation = self.game.reset()
        info = {"score": self.game.score}

        if self.render_mode == "human":
            self.render()

        return observation, info
    
    def reset_bin(self, seed=None, options=None):
        """
        Reset the environment but converts observations
        into a single value from binary observation array
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        observation = self.game.reset()
        bin = np.array2string(observation).strip("[]").replace(" ", "")
        observation = int(bin, 2)
        info = {"score": self.game.score}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step_bin(self, action):
        """
        Execute one step in the environmentbut converts observations
        into a single value from binary observation array
        """
            
        observation, reward, terminated, truncated, info = self.game.take_action(action)

        bin = np.array2string(observation).strip("[]").replace(" ", "")
        observation = int(bin, 2)
        
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def step(self, action):
        """Execute one step in the environment"""
        observation, reward, terminated, truncated, info = self.game.take_action(action)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # Handle pygame events to prevent window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            self.game.render(mode="human")

    def close(self):
        """Close the environment"""
        self.game.close()

from typing import Optional
import gymnasium as gym
import numpy as np

class RandomObsBinaryRewardEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        assert render_mode is None or render_mode in ["human"]

        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(2) # obs = -1 o obs = +1
        self.obs_to_reward = {0: -1.0, 1: 1.0}
        self._terminated = False
        self._obs = None

    def _get_obs(self):
        return np.random.choice([0, 1])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._terminated = False

        self._obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return self._obs, info

    def step(self, action):
        if self._terminated:
            raise RuntimeError("Terminated. reset")

        reward = self.obs_to_reward[self._obs]

        self._terminated = True
        truncated = False

        self._obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return self._obs, reward, self._terminated, truncated, info
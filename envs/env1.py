from typing import Optional
import gymnasium as gym

class ConstantRewardEnv(gym.Env):
  def __init__(self, render_mode=None):
    self.render_mode = render_mode
    assert render_mode is None or render_mode in ["human"]

    self.action_space = gym.spaces.Discrete(1)

    self.observation_space = gym.spaces.Discrete(1)

    self._terminated = False

  def _get_obs(self):
    return 0

  def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
    super().reset(seed=seed)
    self._terminated = False

    observation = self._get_obs()
    info = {}

    if self.render_mode == "human":
      self.render()

    return observation, info

  def step(self, action):
    if self._terminated:
      raise RuntimeError("Terminated. reset")

    reward = 1

    self._terminated = True
    truncated = False

    observation = self._get_obs()
    info = {}

    if self.render_mode == "human":
      self.render()

    return observation, reward, self._terminated, truncated, info
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

class QLearning:
    def __init__(self, env, episodes: int, alpha: float, gamma: float, min_epsilon: float,
        max_epsilon: float = None, decay_rate: float = None, log_dir: str = None):
        
        self.env = env
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        if max_epsilon is None:
            self.max_epsilon = self.min_epsilon
        else:
            self.max_epsilon = max_epsilon
            self.decay_rate = decay_rate
        
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)

        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])


    def get_epsilon(self, episode):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def run_episode(self, episode, seed):
        state, info = self.env.reset(seed=seed)

        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        while not terminated and not truncated:
            epsilon = self.get_epsilon(episode)
            action = self.choose_action(state, epsilon)

            next_state, reward, terminated, truncated, info = self.env.step(action)

            done = terminated or truncated

            target = reward

            if not done:
                target += self.gamma * np.max(self.Q[next_state])

            self.Q[state, action] += self.alpha * (target - self.Q[state, action])

            state = next_state
            total_reward += reward
            steps += 1

        return total_reward, steps
    

    def log_all_q_values(self, episode):
        if not hasattr(self, "writer"):
            return

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n

        for s in range(n_states):
            for a in range(n_actions):
                q_sa = self.Q[s, a]

                self.writer.add_scalar(
                    f"Q-values/Episode/state_{s}_action_{a}",
                    q_sa,
                    episode
                )

                self.writer.add_scalar(
                    f"Q-values/TotalSteps/state_{s}_action_{a}",
                    q_sa,
                    self.total_steps
                )

    def train(self, seed, log_q_values=False):
        total_rewards = []

        for episode in range(self.episodes):
            total_reward, steps = self.run_episode(episode, seed + episode)

            self.total_steps += steps
            total_rewards.append(total_reward)

            if hasattr(self, "writer"):
                
                self.writer.add_scalar("Reward/Episode", total_reward, episode)
                self.writer.add_scalar("Reward/TotalSteps", total_reward, self.total_steps)

                if log_q_values:
                    self.log_all_q_values(episode)

        if hasattr(self, "writer"):
            self.writer.close()

        return total_rewards

    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        np.save(path, self.Q)

    def load_model(self, path):
        self.Q = np.load(path)
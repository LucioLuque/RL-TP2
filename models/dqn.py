import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import gymnasium as gym
import torch.nn.functional as F
from tqdm.auto import tqdm

from q_network import QNetwork


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.seen = 0

    def update(self, transition):
        if len(self.buffer) < self.size:
            self.buffer.append(transition)
        else:
            index = self.seen % self.size
            self.buffer[index] = transition
        self.seen += 1

    def get_transition(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

class DQN:
    def __init__(self, env, episodes: int, buffer_size: int, max_steps: int, gamma: float, lr: float,
        target_update_freq: int, min_epsilon: float, max_epsilon: float, decay_rate: float, log_dir: str = None):
        self.env = env
        self.episodes = episodes

        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.max_steps = max_steps
        self.total_steps = 0

        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = decay_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if isinstance(env.observation_space, gym.spaces.Discrete):
            input_dim = env.observation_space.n
        else:
            input_dim = int(np.prod(env.observation_space.shape))
            
        self.q_net = QNetwork(input_dim=input_dim, output_dim=env.action_space.n).to(self.device)
        self.target_q_net = QNetwork(input_dim=input_dim, output_dim=env.action_space.n).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
        self.updates = 0

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
                
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)

    def get_epsilon(self, episode):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def obs_to_tensor(self, states):
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            states = torch.tensor(states, dtype=torch.long, device=self.device)
            return F.one_hot(states, num_classes=self.env.observation_space.n).float()
        else:
            return torch.as_tensor(states, dtype=torch.float32, device=self.device)
        
    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        
        state_tensor = self.obs_to_tensor([state])

        return self.q_net.get_action(state_tensor)

    def update(self, batch_size):
        batch = self.replay_buffer.get_transition(batch_size)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = self.obs_to_tensor(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        next_states_tensor = self.obs_to_tensor(next_states)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
        
    def run_episode(self, episode, batch_size, seed = None):
        if seed is not None:
            state, info = self.env.reset(seed = seed + episode)
        else:
            state, info = self.env.reset()

        total_reward = 0.0
        losses = []
        epsilon = self.get_epsilon(episode)

        for step in range(self.max_steps):
            action = self.choose_action(state, epsilon)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.update((state, action, reward, next_state, done))

            loss = self.update(batch_size)

            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        mean_loss = np.mean(losses) if losses else 0.0
        mean_reward = total_reward / (step + 1)

        return total_reward, mean_reward, mean_loss, step + 1
    
    def log_all_q_values(self, episode):
        if not hasattr(self, "writer"):
            return

        n_actions = self.env.action_space.n

        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            n_states = self.env.observation_space.n
            states = list(range(n_states))
            states_tensor = self.obs_to_tensor(states)
            state_names = [f"state_{s}" for s in states]

        else:
            return

        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(states_tensor).detach().cpu().numpy()
        self.q_net.train()

        for s_idx, state_name in enumerate(state_names):
            for a in range(n_actions):
                q_sa = q_values[s_idx, a]

                self.writer.add_scalar(
                    f"Q-values/Episode/{state_name}_action_{a}",
                    q_sa,
                    episode
                )
    
    def train(self, batch_size = 32, seed = None, log_q_values = False):
        bar = tqdm(range(self.episodes), desc="Training DQN")
        for episode in bar:
            total_reward, reward_per_step, mean_loss, episode_steps = self.run_episode(episode, batch_size, seed)
            self.total_steps += episode_steps
            if hasattr(self, 'writer'):
                self.writer.add_scalar('Reward/Episode', total_reward, episode)
                self.writer.add_scalar("Reward/TotalSteps", total_reward, self.total_steps)


                self.writer.add_scalar('Loss/Episode', mean_loss, episode)
                self.writer.add_scalar('Mean Reward per step/Episode', reward_per_step, episode)

                if log_q_values:
                    self.log_all_q_values(episode)
            
            bar.set_postfix({'Reward': total_reward, 'Mean Reward': reward_per_step, 'Loss': mean_loss})


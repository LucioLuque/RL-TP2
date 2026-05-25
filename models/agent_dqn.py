import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import gymnasium as gym
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import deque
import pickle

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
    
    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)

        self.seen = len(self.buffer)

class AgentDQN:
    def __init__(self, env, episodes: int, buffer_size: int, max_steps: int, gamma: float, lr: float,
        tau: float, min_epsilon: float, max_epsilon: float, decay_rate: float, prefill_episodes: int = 20, prefill_epsilon: float = 0.05, n_step: int = 1, log_dir: str = None, huber: bool = False):
        self.env = env
        self.episodes = episodes

        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.max_steps = max_steps
        self.total_steps = 0

        self.gamma = gamma
        self.tau = tau

        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = decay_rate

        self.prefill_episodes = prefill_episodes
        self.prefill_epsilon = prefill_epsilon

        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        
        self.huber = huber

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
            #ddqn
            next_actions = self.q_net(next_states_tensor).argmax(1)
            next_q_values = self.target_q_net(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_q_values = rewards_tensor + (self.gamma ** self.n_step) * next_q_values * (1 - dones_tensor)
        
        if self.huber:
            loss = F.smooth_l1_loss(q_values, target_q_values)
        else:
            loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        
        loss.backward()
        if self.huber:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()
    
    def reward_shaping(self, state, reward, terminated):
        reward += 3 * abs(state[1]) #velocity

        if terminated:
            reward += 10

        return reward
    
    def prefill_replay_buffer(self, seed=None):
        successful = 0
        attempts = 0
        max_attempts = 1000
        while successful < self.prefill_episodes and attempts < max_attempts:
            if seed is not None:
                state, _ = self.env.reset(seed=seed + attempts)
            else:
                state, _ = self.env.reset()
            attempts += 1
            self.n_step_buffer.clear()
            episode_transitions = []
            done = False
            while not done:
                position, velocity = state

                # policy
                if velocity > 0:
                    action = 2  # push right
                else:
                    action = 0  # push left

                if np.random.rand() < self.prefill_epsilon:
                    action = self.env.action_space.sample()

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                reward = self.reward_shaping(next_state, reward, terminated)

                self.n_step_buffer.append((state, action, reward, next_state, done))

                if len(self.n_step_buffer) == self.n_step:
                    transition = self.compute_n_step_transition()

                    episode_transitions.append(transition)

                    self.n_step_buffer.popleft()

                state = next_state

                if done:
                    while len(self.n_step_buffer) > 0:
                        transition = self.compute_n_step_transition()
                        episode_transitions.append(transition)
                        self.n_step_buffer.popleft()

                    if terminated: #only succesful episodes
                        for transition in episode_transitions:
                            self.replay_buffer.update(transition)
                        successful += 1
                    break

        print(f"Prefilled replay buffer with {successful} successful episodes in {attempts} attempts.")
    
    def compute_n_step_transition(self):
        reward, next_state, done = 0, None, False
        for i, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * r

            next_state = ns
            done = d

            if done:
                break

        state, action = self.n_step_buffer[0][:2]

        return state, action, reward, next_state, done
        
    def run_episode(self, episode, batch_size, seed = None):
        if seed is not None:
            state, info = self.env.reset(seed = seed + episode)
        else:
            state, info = self.env.reset()

        self.n_step_buffer.clear()

        total_reward = 0.0
        pure_reward_total = 0.0
        losses = []
        epsilon = self.get_epsilon(episode)

        for step in range(self.max_steps):
            self.total_steps += 1
            action = self.choose_action(state, epsilon)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            pure_reward = reward
            pure_reward_total += pure_reward
            reward = self.reward_shaping(next_state, reward, terminated)

            self.n_step_buffer.append((state, action, reward, next_state, done))
            if len(self.n_step_buffer) == self.n_step:
                transition = self.compute_n_step_transition()
                self.replay_buffer.update(transition)
                self.n_step_buffer.popleft()

            loss = None
            if self.total_steps % 4 == 0: # update every 4 steps
                loss = self.update(batch_size)

            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                while len(self.n_step_buffer) > 0:
                    transition = self.compute_n_step_transition()
                    self.replay_buffer.update(transition)

                    self.n_step_buffer.popleft()
                break

        mean_loss = np.mean(losses) if losses else 0.0
        mean_reward = total_reward / (step + 1)

        return total_reward, mean_reward, mean_loss, pure_reward_total
    
    def train(self, batch_size = 32, seed = None, prefill_path=None):

        if prefill_path is not None and os.path.exists(prefill_path):
            self.replay_buffer.load(prefill_path)
        else:
            self.prefill_replay_buffer(seed)
            if prefill_path is not None:
                self.replay_buffer.save(prefill_path)
        
        bar = tqdm(range(self.episodes), desc="Training DQN")
        for episode in bar:
            total_reward, reward_per_step, mean_loss, pure_reward_total = self.run_episode(episode, batch_size, seed)

            if hasattr(self, 'writer'):
                self.writer.add_scalar('Reward/Episode', total_reward, episode)
                self.writer.add_scalar("Reward/TotalSteps", total_reward, self.total_steps)
                self.writer.add_scalar('Reward/Pure/Episode', pure_reward_total, episode)

                self.writer.add_scalar('Loss/Episode', mean_loss, episode)
                self.writer.add_scalar('Mean Reward per step/Episode', reward_per_step, episode)
            
            bar.set_postfix({'Reward': total_reward, 'Mean Reward': reward_per_step, 'Loss': mean_loss})
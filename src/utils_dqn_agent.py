import os
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import torch
import gymnasium as gym
import torch.nn.functional as F
import optuna

from utils import deterministic, load_from_tensorboard
from agent_dqn import AgentDQN
from q_network import QNetwork

def train_dqn_agent(experiment_folder, env, episodes, buffer_size, max_steps, gamma, lr, tau, min_epsilon, max_epsilon, decay_rate, batch_size, prefill_episodes, prefill_epsilon, n_step, seed=42, save=True, log_q_values=False, trial = None, prefill_path = None):

    run_folder = f"../runs/agent_dqn/{experiment_folder}"
    experiment = f"epi_{episodes}_buf_{buffer_size}_steps_{max_steps}_g_{gamma}_lr_{lr}_tau_{tau}_eps_{min_epsilon}_max_eps_{max_epsilon}_decay_{decay_rate}_batch_{batch_size}_n_{n_step}_pepi_{prefill_episodes}_peps_{prefill_epsilon}"
    path = f"{run_folder}/{experiment}"

    model_path = f"../models/saved_models/agent_dqn/{experiment_folder}/{experiment}.npy"

    if os.path.exists(path):
        print(f"Experiment {run_folder}/{experiment} already exists.")
        q_net_state_dict = torch.load(model_path)
        return model_path, q_net_state_dict, None
    
    deterministic(seed)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    log_dir = path if save else None
    
    dqn_model = AgentDQN(env, episodes= episodes, buffer_size = buffer_size,
            max_steps = max_steps, gamma = gamma, lr = lr, tau = tau,
            min_epsilon = min_epsilon, max_epsilon = max_epsilon, decay_rate = decay_rate,
            prefill_episodes=prefill_episodes, prefill_epsilon=prefill_epsilon, n_step=n_step,
            log_dir = log_dir)
    training_results = dqn_model.train(batch_size = batch_size, seed = seed,
                                    log_q_values = log_q_values, trial = trial, prefill_path = prefill_path)
    if save:
        dqn_model.q_net.save(model_path)

    return model_path, dqn_model.q_net.state_dict(), training_results

def obs_to_tensor(state, env, device):
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return F.one_hot(
            torch.tensor(state, dtype=torch.long),
            num_classes=env.observation_space.n
        ).float().unsqueeze(0).to(device)
    else:
        return torch.as_tensor(
            state, dtype=torch.float32
        ).reshape(1, -1).to(device)


def evaluate_dqn(path, env, episodes=100, seed=42):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    if isinstance(env.observation_space, gym.spaces.Discrete):
        input_dim = env.observation_space.n
    else:
        input_dim = int(np.prod(env.observation_space.shape))

    q_net = QNetwork(input_dim, env.action_space.n).to(device)
    q_net.load(path)

    episode_rewards = []
    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False
        reward_ep = 0.0

        while not terminated and not truncated:
            state_tensor = obs_to_tensor([state], env, device)
            action = q_net.get_action(state_tensor)
            state, reward, terminated, truncated, info = env.step(action)
            reward_ep += reward

        episode_rewards.append(reward_ep)

    mean_reward = float(np.mean(episode_rewards))
    std = float(np.std(episode_rewards))

    return mean_reward, std

# def evaluate_dqn_model(path, env, model_params, episodes=100, seed=42):  
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
#     if isinstance(env.observation_space, gym.spaces.Discrete):
#         input_dim = env.observation_space.n
#     else:
#         input_dim = int(np.prod(env.observation_space.shape))

#     q_net = QNetwork(input_dim, env.action_space.n).to(device)
#     q_net.load_state_dict(model_params)
#     q_net.eval()

#     successes = 0
#     rewards = 0.0
#     for episode in range(episodes):
#         state, info = env.reset(seed=seed + episode)

#         terminated = False
#         truncated = False
#         reward_ep = 0.0

#         while not terminated and not truncated:
#             state_tensor = obs_to_tensor([state], env, device)
#             action = q_net.get_action(state_tensor)
#             with torch.no_grad():
#                     q_values = q_net(state_tensor)
#             state, reward, terminated, truncated, info = env.step(action)
#             reward_ep += reward
#         if reward_ep == 500:
#             successes += 1
        
#         rewards += reward_ep

#     return successes / episodes, rewards / episodes

def record_experiment(experiment_path, experiment_name, env, num_episodes=1, epsilon=0.0, seed=42):
    deterministic(seed=seed)
    video_folder = os.path.join("..", "videos")
    os.makedirs(video_folder, exist_ok=True)
    
    env = RecordVideo(
        env, 
        video_folder=video_folder,
        episode_trigger=lambda ep: True,  
        name_prefix=experiment_name
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    if isinstance(env.observation_space, gym.spaces.Discrete):
        input_dim = env.observation_space.n
    else:
        input_dim = int(np.prod(env.observation_space.shape))

    q_net = QNetwork(input_dim, env.action_space.n).to(device)
    q_net.load(experiment_path)

    for episode in range(num_episodes):
        state, info = env.reset(seed=seed if episode == 0 else None)
        terminated = False
        total_reward = 0

        while not terminated:
            state_tensor = obs_to_tensor([state], env, device)
            action = q_net.get_action(state_tensor)
            state, reward, terminated, truncated, info = env.step(action)
            terminated = terminated or truncated
            total_reward += reward

        print(f"Episode {episode+1} | Total reward: {total_reward}")

    env.close()
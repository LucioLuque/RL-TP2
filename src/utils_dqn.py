import os
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import torch
import gymnasium as gym
import torch.nn.functional as F

from utils import deterministic, load_from_tensorboard
from dqn import DQN
from q_network import QNetwork

def train_dqn(experiment_folder, env, episodes, buffer_size, max_steps, gamma, lr, target_update_freq, min_epsilon, max_epsilon, decay_rate, batch_size,seed=42, save=True, log_q_values=False):

    run_folder = f"../runs/dqn/{experiment_folder}"
    experiment = f"epi_{episodes}_buf_{buffer_size}_steps_{max_steps}_g_{gamma}_lr_{lr}_target_{target_update_freq}_eps_{min_epsilon}_max_eps_{max_epsilon}_decay_{decay_rate}_batch_{batch_size}"
    path = f"{run_folder}/{experiment}"

    model_path = f"../models/saved_models/dqn/{experiment_folder}/{experiment}.npy"

    if os.path.exists(path):
        print(f"Experiment {run_folder}/{experiment} already exists.")
        q_net_state_dict = torch.load(model_path)
        return model_path, q_net_state_dict
    
    deterministic(seed)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    log_dir = path if save else None
    
    dqn_model = DQN(env, episodes, buffer_size, max_steps, gamma, lr, target_update_freq, min_epsilon, max_epsilon, decay_rate, log_dir)
    dqn_model.train(batch_size, seed, log_q_values)
    if save:
        dqn_model.q_net.save(model_path)

    return model_path, dqn_model.q_net.state_dict()


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

def evaluate_dqn_env1(path, env, episodes=100, seed=42):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    q_net = QNetwork(env.observation_space.n, env.action_space.n).to(device)
    q_net.load(path)

    successes = 0

    unique_q_values = set()

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            state_tensor = obs_to_tensor([state], env, device)
            action = q_net.get_action(state_tensor)
            with torch.no_grad():
                    q_values = q_net(state_tensor)
                    unique_q_values.update(q_values.cpu().numpy().flatten())
            state, reward, terminated, truncated, info = env.step(action)

        if reward == 1:
            successes += 1

    return successes / episodes, unique_q_values

def evaluate_dqn_env2(path, env, episodes=100, seed=42):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(env.observation_space, gym.spaces.Discrete):
        input_dim = env.observation_space.n
    else:
        input_dim = int(np.prod(env.observation_space.shape))
    q_net = QNetwork(input_dim, env.action_space.n).to(device)
    q_net.load(path)
    q_net.eval()

    obs_q = {
        -1.0: set(),
        1.0: set()
    }

    total_error = 0.0
    total_steps = 0

    def to_semantic_obs(state_value):
        return -1.0 if int(state_value) == 0 else 1.0

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            state_tensor = obs_to_tensor([state], env, device)

            with torch.no_grad():
                q_values = q_net(state_tensor)
            action = q_values.argmax(dim=1).item()

            obs_value = to_semantic_obs(np.array(state).reshape(-1)[0])
            q_value = q_values.cpu().numpy().flatten()[0].item()

            obs_q[obs_value].add(round(q_value, 4))
            
            total_error += (q_value - obs_value) ** 2
            state, reward, terminated, truncated, info = env.step(action)

    mse = total_error / episodes

    return mse, obs_q


def evaluate_dqn_env3(path, env, gamma, episodes=100, seed=42, tol=0.1): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(env.observation_space, gym.spaces.Discrete):
        input_dim = env.observation_space.n
    else:
        input_dim = int(np.prod(env.observation_space.shape))


    q_net = QNetwork(input_dim, env.action_space.n).to(device)
    q_net.load(path)
    q_net.eval()

    expected_q= {
        0: gamma,
        1: 1.0
    }

    obs_q = {0: set(), 1: set()}

    total_error = 0.0
    total_predictions = 0
    correct_predictions = 0

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            state_tensor = obs_to_tensor([state], env, device)

            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = q_values.argmax(dim=1).item()

            obs_value = int(state)
            q_value = q_values.cpu().numpy().flatten()[0].item()

            obs_q[obs_value].add(round(q_value, 4))

            target_value = expected_q[obs_value]

            total_error += (q_value - target_value) ** 2
            total_predictions += 1

            if abs(q_value - target_value) <= tol:
                correct_predictions += 1

            state, reward, terminated, truncated, info = env.step(action)

    mse = total_error / total_predictions
    success_rate = correct_predictions / total_predictions

    return success_rate, mse, obs_q


def evaluate_dqn(path, env, episodes=100, seed=42):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    if isinstance(env.observation_space, gym.spaces.Discrete):
        input_dim = env.observation_space.n
    else:
        input_dim = int(np.prod(env.observation_space.shape))

    q_net = QNetwork(input_dim, env.action_space.n).to(device)
    q_net.load(path)

    successes = 0
    rewards = 0.0
    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False
        reward_ep = 0.0

        while not terminated and not truncated:
            state_tensor = obs_to_tensor([state], env, device)
            action = q_net.get_action(state_tensor)
            with torch.no_grad():
                    q_values = q_net(state_tensor)
            state, reward, terminated, truncated, info = env.step(action)
            reward_ep += reward
        if reward_ep == 500:
            successes += 1
        
        rewards += reward_ep

    return successes / episodes, rewards / episodes


def evaluate_dqn_model(path, env, model_params, episodes=100, seed=42):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    if isinstance(env.observation_space, gym.spaces.Discrete):
        input_dim = env.observation_space.n
    else:
        input_dim = int(np.prod(env.observation_space.shape))

    q_net = QNetwork(input_dim, env.action_space.n).to(device)
    q_net.load_state_dict(model_params)
    q_net.eval()

    successes = 0
    rewards = 0.0
    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False
        reward_ep = 0.0

        while not terminated and not truncated:
            state_tensor = obs_to_tensor([state], env, device)
            action = q_net.get_action(state_tensor)
            with torch.no_grad():
                    q_values = q_net(state_tensor)
            state, reward, terminated, truncated, info = env.step(action)
            reward_ep += reward
        if reward_ep == 500:
            successes += 1
        
        rewards += reward_ep

    return successes / episodes, rewards / episodes

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
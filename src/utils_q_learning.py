import os
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

from utils import deterministic, load_from_tensorboard
from q_learning import QLearning

def train_experiment(experiment_folder, env, episodes, alpha, gamma, min_epsilon, max_epsilon=None, decay_rate=None, seed=42, save=True):

    run_folder = f"../runs/{experiment_folder}"
    experiment = f"epi_{episodes}_a_{alpha}_g_{gamma}_eps_{min_epsilon}"
    if max_epsilon is not None and decay_rate is not None:
        experiment += f"_max_eps_{max_epsilon}_decay_{decay_rate}"
    path = f"{run_folder}/{experiment}"

    model_path = f"../models/saved_models/{experiment_folder}/{experiment}.npy"

    if os.path.exists(path):
        print(f"Experiment {run_folder}/{experiment} already exists.")
        Q = np.load(model_path)
        return model_path, Q
    
    deterministic(seed)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    log_dir = path if save else None

    q_learning_model = QLearning(env, episodes, alpha, gamma, min_epsilon, max_epsilon, decay_rate, log_dir)
    q_learning_model.train(seed)
    if save:
        q_learning_model.save_model(model_path)

    return model_path, q_learning_model.Q.copy()

def evaluate_Q(path, env, episodes=100, epsilon=0.0, seed=42):    
    q_learning_model = QLearning(env, 0, 0, 0, 0, log_dir=None)
    q_learning_model.load_model(path)

    successes = 0

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = q_learning_model.choose_action(state, epsilon)
            state, reward, terminated, truncated, info = env.step(action)

        if reward == 1:
            successes += 1

    return successes / episodes

def evaluate_Q_model(Q, env, episodes=100, epsilon=0.0, seed=42):    
    q_learning_model = QLearning(env, 0, 0, 0, 0, log_dir=None)
    q_learning_model.Q = Q

    successes = 0

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = q_learning_model.choose_action(state, epsilon)
            state, reward, terminated, truncated, info = env.step(action)

        if reward == 1:
            successes += 1

    return successes / episodes

def search_hyperparameters(experiment_folder, env, episodes, gamma, alphas, epsilons, seed):
    best_a_eps = {alpha: None for alpha in alphas}
    best_rates = {alpha: 0.0 for alpha in alphas}
    bar = tqdm(total=len(alphas) * len(epsilons), desc="Searching hyper")
    for i, alpha in enumerate(alphas):
        best_success_rate = 0.0
        if i > 0:
            print(f"Best epsilon for alpha {alphas[i-1]}: {best_a_eps[alphas[i-1]]} with success rate {best_rates[alphas[i-1]]:.2f}")
        for eps in epsilons:
            path, Q = train_experiment(experiment_folder, env, episodes, alpha, gamma, eps, seed, save=False)
            
            success_rate = evaluate_Q_model(Q, env, episodes=100, epsilon=0.0, seed=seed)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_a_eps[alpha] = eps
                best_rates[alpha] = success_rate

            bar.set_postfix({"a": alpha, "eps": eps, "succ_rate": success_rate})
            bar.update(1)
    bar.close()
    return best_a_eps


def search_eps_decay(experiment_folder, env, episodes, gamma, alpha, min_epsilons, max_epsilons, decays_rate, seed):
    min_max_pairs = [(min_eps, max_eps) for min_eps in min_epsilons for max_eps in max_epsilons if min_eps < max_eps]

    best_max_decay = {min_eps: None for min_eps in min_epsilons}
    best_rates = {min_eps: 0.0 for min_eps in min_epsilons}
    total_iterations = len(min_max_pairs) * len(decays_rate)
    print(f"Total hyperparameter combinations to evaluate: {total_iterations}")
    bar = tqdm(total=total_iterations, desc="Searching hyper")

    global_best = { "min_epsilon": None, "max_epsilon": None, "decay_rate": None, "success_rate": 0.0}
    for i, (min_epsilon, max_epsilon) in enumerate(min_max_pairs):
        for decay_rate in decays_rate:

            path, Q = train_experiment(experiment_folder, env, episodes, alpha, gamma, min_epsilon, max_epsilon, decay_rate, seed, save=False)
            
            success_rate = evaluate_Q_model(Q, env, episodes=100, epsilon=0.0, seed=seed)
            if success_rate > best_rates[min_epsilon]:
                best_max_decay[min_epsilon] = (max_epsilon, decay_rate)
                best_rates[min_epsilon] = success_rate

            if success_rate > global_best["success_rate"]:
                global_best = {
                    "min_epsilon": min_epsilon,
                    "max_epsilon": max_epsilon,
                    "decay_rate": decay_rate,
                    "success_rate": success_rate,
                }

            bar.set_postfix({"min_eps": min_epsilon, "max_eps": max_epsilon, "decay": decay_rate, "succ_rate": success_rate, "best": global_best})
            bar.update(1)

    bar.close()

    return best_max_decay, global_best

def plot_mean_rewards(folder_name, window=100):
    path = Path(f"../runs/{folder_name}")
    episodes, rewards = load_from_tensorboard(path, "Reward")

    mean_rewards = [np.mean(rewards[i:i+window]) for i in range(0, len(rewards), window)]

    x_values = [min(i + window - 1, len(rewards) - 1) for i in range(0, len(rewards), window)]

    fig = plt.figure(figsize=(10, 4))
    plt.plot(x_values, mean_rewards, marker='o', label="Mean Reward (100 episodes)")
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Mean Reward", fontsize=16)
    xticks = list(range(0, len(rewards), 100))

    if len(rewards) - 1 not in xticks:
        xticks.append(len(rewards) - 1)
    plt.xticks(xticks, rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize = 16)
    plt.show()

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
    

    q_learning_model = QLearning(env, 0, 0, 0, 0, log_dir=None)
    q_learning_model.load_model(experiment_path)

    for episode in range(num_episodes):
        state, info = env.reset(seed=seed if episode == 0 else None)
        terminated = False
        total_reward = 0

        while not terminated:
            action = q_learning_model.choose_action(state, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            terminated = terminated or truncated
            total_reward += reward

        print(f"Episode {episode+1} | Total reward: {total_reward}")

    env.close()



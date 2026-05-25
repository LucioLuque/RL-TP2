import os
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from matplotlib.patches import Polygon, Rectangle
import gymnasium as gym

from utils import deterministic, load_from_tensorboard
from q_learning import QLearning

def train_q_learning(experiment_folder, env, episodes, alpha, gamma, min_epsilon, max_epsilon=None, decay_rate=None, seed=42, save=True, log_q_values=False):
    run_folder = f"../runs/q_learning/{experiment_folder}"
    experiment = f"epi_{episodes}_a_{alpha}_g_{gamma}_eps_{min_epsilon}"
    if max_epsilon is not None and decay_rate is not None:
        experiment += f"_max_eps_{max_epsilon}_decay_{decay_rate}"
    path = f"{run_folder}/{experiment}"

    model_path = f"../models/saved_models/q_learning/{experiment_folder}/{experiment}.npy"

    if os.path.exists(path):
        print(f"Experiment {run_folder}/{experiment} already exists.")
        Q = np.load(model_path)
        return model_path, Q
    
    deterministic(seed)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    log_dir = path if save else None

    q_learning_model = QLearning(env, episodes, alpha, gamma, min_epsilon, max_epsilon, decay_rate, log_dir)
    q_learning_model.train(seed=seed, log_q_values=log_q_values)
    if save:
        q_learning_model.save_model(model_path)

    return model_path, q_learning_model.Q.copy()

def evaluate_Q(path, env, episodes=100, epsilon=0.0, seed=42):    
    q_learning_model = QLearning(env, 0, 0, 0, 0, log_dir=None)
    q_learning_model.load_model(path)

    successes = 0

    truncateds = 0

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = q_learning_model.choose_action(state, epsilon)
            state, reward, terminated, truncated, info = env.step(action)
            if truncated:
                truncateds += 1
                break
        if reward == 1:
            successes += 1

    return successes / episodes, truncateds

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
            path, Q = train_q_learning(experiment_folder, env, episodes, alpha, gamma, eps, seed=seed, save=False)
            
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
    bests = []
    for i, (min_epsilon, max_epsilon) in enumerate(min_max_pairs):
        for decay_rate in decays_rate:

            path, Q = train_q_learning(experiment_folder, env, episodes, alpha, gamma, min_epsilon, max_epsilon, decay_rate, seed=seed, save=False)
            
            success_rate = evaluate_Q_model(Q, env, episodes=100, epsilon=0.0, seed=seed)
            if success_rate > best_rates[min_epsilon]:
                best_max_decay[min_epsilon] = (max_epsilon, decay_rate, success_rate)
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

def plot_mean_rewards(folder_name, img_path, window=100):
    path = Path(f"../runs/{folder_name}")
    episodes, rewards = load_from_tensorboard(path, "Reward/Episode")

    mean_rewards = [np.mean(rewards[i:i+window]) for i in range(0, len(rewards), window)]

    x_values = [min(i + window - 1, len(rewards) - 1) for i in range(0, len(rewards), window)]

    fig = plt.figure(figsize=(10, 4))
    plt.plot(x_values, mean_rewards, marker='o', label=f"Mean Reward ({window} episodes)")
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Mean Reward", fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize = 16)

    path = Path(img_path)
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)
    plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()

def record_experiment(experiment_path, experiment_name, env, num_episodes=1, epsilon=0.0, seed=42):
    deterministic(seed=seed)
    video_folder = os.path.join("..", "videos")
    os.makedirs(video_folder, exist_ok=True)
    
    env = RecordVideo(env,  video_folder=video_folder, episode_trigger=lambda ep: True, name_prefix=experiment_name)

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

def softmax(x, temperature=1.0):
    x = np.array(x, dtype=np.float64)

    x = x / temperature
    x = x - np.max(x)
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)

def q_to_action_probs(q_values, temperature=0.2):
    q_values = np.array(q_values, dtype=np.float64)
    return softmax(q_values, temperature=temperature)

def setup_axis(ax, frame, n_cols, n_rows, subtitle):
    ax.imshow(frame, extent=[0, n_cols, n_rows, 0], interpolation="nearest", zorder=0)

    for row in range(n_rows):
        for col in range(n_cols):
            rect = Rectangle((col, row),1,1, fill=False, edgecolor="black", linewidth=1.2, zorder=5)
            ax.add_patch(rect)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels([str(i) for i in range(n_cols)])
    ax.set_yticklabels([str(i) for i in range(n_rows)])

    ax.set_title(subtitle)

def plot_frozenlake_policy_comparison(Q, temperature=0.2, save_path=None, policy_alpha=0.45):
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=True)

    state, info = env.reset()
    
    Q = np.array(Q, dtype=np.float64)

    frame = env.render()
    desc = env.unwrapped.desc.astype(str)

    n_rows = 4
    n_cols = 4

    action_names = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    probs_matrix = np.zeros_like(Q, dtype=np.float64)
    greedy_actions = np.argmax(Q, axis=1)

    for s in range(16):
        probs_matrix[s] = q_to_action_probs(Q[s], temperature=temperature)

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.15)

    ax_probs = fig.add_subplot(gs[0, 0])
    ax_greedy = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    setup_axis(ax_probs, frame, n_cols, n_rows, "Action Probabilities") 
    setup_axis(ax_greedy, frame, n_cols, n_rows, "Greedy Policy")

    skip_cell_types=("H", "G")
    for s in range(16):
        row = s // n_cols
        col = s % n_cols
        cell_type = desc[row, col]

        if cell_type in skip_cell_types:
            continue

        x0 = col
        y0 = row
        x1 = col + 1
        y1 = row + 1
        cx = col + 0.5
        cy = row + 0.5

        triangles = {
            0: [(x0, y0), (x0, y1), (cx, cy)],  # left
            1: [(x0, y1), (x1, y1), (cx, cy)],  # down
            2: [(x1, y0), (x1, y1), (cx, cy)],  # right
            3: [(x0, y0), (x1, y0), (cx, cy)],  # up
        }

        for action in range(4):
            prob = probs_matrix[s, action]

            polygon = Polygon(triangles[action], closed=True, facecolor=cmap(norm(prob)), edgecolor="black",
                linewidth=0.6, alpha=policy_alpha, zorder=2)
            ax_probs.add_patch(polygon)

        text_positions = {0: (x0 + 0.18, cy), 1: (cx, y1 - 0.18), 2: (x1 - 0.18, cy), 3: (cx, y0 + 0.18)}

        for action in range(4):
            tx, ty = text_positions[action]
            ax_probs.text(tx, ty, f"{action_names[action]}\n{probs_matrix[s, action]:.2f}",
                ha="center", va="center", fontsize=8, color="white", zorder=6,
                bbox=dict(facecolor="black", alpha=0.18, edgecolor="none", pad=0.4))

    for s in range(16):
        row = s // n_cols
        col = s % n_cols
        cell_type = desc[row, col]

        if cell_type in skip_cell_types:
            continue

        best_action = greedy_actions[s]

        cx = col + 0.5
        cy = row + 0.5

        ax_greedy.text(cx, cy, action_names[best_action], ha="center", va="center", fontsize=28,
            fontweight="bold", color="black", zorder=6,
            bbox=dict(facecolor="white", alpha=0.45, edgecolor="none", boxstyle="round,pad=0.2"))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Probabilidad de acción")
    fig.subplots_adjust(top=0.88)

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
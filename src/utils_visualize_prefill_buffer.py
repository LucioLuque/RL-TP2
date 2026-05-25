import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image as PILImage
from IPython.display import display, Image as IPyImage

def load_prefill_buffer(prefill_path):
    with open(prefill_path, "rb") as f:
        buffer = pickle.load(f)

    print(f"Loaded {len(buffer)} transitions")

    return buffer

def split_buffer_into_episodes(buffer):
    episodes = []
    current_episode = []

    for i, transition in enumerate(buffer):
        current_episode.append(transition)

        done = bool(transition[-1])

        next_is_done = (i + 1 < len(buffer) and bool(buffer[i + 1][-1]))

        if done and not next_is_done:
            episodes.append(current_episode)
            current_episode = []

    print(f"Recovered {len(episodes)} episodes")

    return episodes

def inspect_episodes(episodes, n=5):
    lengths = [len(ep) for ep in episodes]
    print(f"Episodes: {len(episodes)}")
    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}\n")

    for i in range(min(n, len(episodes))):
        print(f"Episode {i}: {len(episodes[i])} transitions")

def plot_episode_trajectory(episode):
    positions = []
    velocities = []
    actions = []

    for transition in episode:
        state, action, reward, next_state, done = transition

        positions.append(state[0])
        velocities.append(state[1])
        actions.append(action)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(positions)
    axes[0].set_ylabel("Position", fontsize=16)
    axes[0].grid()

    axes[1].plot(velocities)
    axes[1].set_ylabel("Velocity", fontsize=16)
    axes[1].grid()

    axes[2].step(range(len(actions)), actions, where="post")
    axes[2].set_ylabel("Action", fontsize=16)
    axes[2].set_xlabel("Step", fontsize=16)
    axes[2].grid()

    plt.tight_layout()
    plt.show()

def render_episode_frames(episode, env_name="MountainCar-v0"):

    env = gym.make(env_name, render_mode="rgb_array")
    env.reset()

    frames = []

    sampled_episode = episode[::1]

    for transition in sampled_episode:
        state = np.asarray(transition[0], dtype=np.float32)

        env.unwrapped.state = state.copy()
        frame = env.render()
        frames.append(frame)

    last_next_state = np.asarray(episode[-1][3], dtype=np.float32)
    env.unwrapped.state = last_next_state.copy()
    frames.append(env.render())

    env.close()
    return frames

def display_episode_gif(episode, env_name="MountainCar-v0"):
    frames = render_episode_frames(episode, env_name=env_name)
    duration_ms = int(1000 / 25)
    pil_frames = [PILImage.fromarray(frame) for frame in frames]
    buffer = BytesIO()
    pil_frames[0].save(buffer, format="GIF", save_all=True,
        append_images=pil_frames[1:], duration=duration_ms, loop=0)

    display(IPyImage(data=buffer.getvalue(), width=500))

def generate_expert_episode(expert_model, seed=42, max_steps=200, plot=True):
    env = gym.make("MountainCar-v0", render_mode=None)
    state, _ = env.reset(seed=seed)
    episode = []
    terminated = False
    truncated = False

    total_reward = 0.0

    for step in range(max_steps):

        if np.random.rand() < 0.05:
            action = env.action_space.sample()
        else:
            action, _ = expert_model.predict(
                state,
                deterministic=True
            )

        action = int(np.asarray(action).item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        episode.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                action,
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                done,
            )
        )

        state = next_state

        if done:
            break

    env.close()

    if plot:
        plot_episode_trajectory(episode)

    return episode
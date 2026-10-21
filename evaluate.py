"""Score the best DQN and PPO models, plot reward curves, and record rollout videos."""
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # save plots to a file; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from train import ENV_ID

ALGOS = {"dqn": DQN, "ppo": PPO}
N_EPISODES = 100       # fresh episodes per model
EVAL_SEED = 1000       # differs from the seeds used in training and in-training evaluation
SOLVED_SCORE = 200     # common LunarLander "solved" threshold
VIDEOS_PER_ALGO = 1
SMOOTH_WINDOW = 5      # evaluations averaged in the smoothed curve


def load_best_model(algo):
    """Load the best checkpoint saved by EvalCallback during training."""
    return ALGOS[algo].load(f"models/{algo}/best_model", device="cpu")


def score_model(model):
    """Run N_EPISODES fresh episodes and return summary stats."""
    env = make_vec_env(ENV_ID, n_envs=1, seed=EVAL_SEED)
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=N_EPISODES,
                                       deterministic=True, return_episode_rewards=True)
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "pct_episodes_solved": 100 * np.mean(np.array(rewards) >= SOLVED_SCORE),
        "mean_episode_length": np.mean(lengths),
    }


def record_videos(model, algo):
    """Record rollouts of the model to videos/<algo>-episode-<n>.mp4."""
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos", name_prefix=algo, episode_trigger=lambda episode: True)
    for episode in range(VIDEOS_PER_ALGO):
        obs, _ = env.reset(seed=EVAL_SEED + episode)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
    env.close()  # finishes writing the last video


def plot_reward_curves():
    """Plot evaluation reward against environment steps for both algorithms."""
    plt.figure(figsize=(8, 5))
    for algo in ALGOS:
        data = np.load(f"logs/{algo}/evaluations.npz")
        mean_reward = data["results"].mean(axis=1)  # average over the episodes of each evaluation
        smoothed = pd.Series(mean_reward).rolling(SMOOTH_WINDOW, min_periods=1).mean()
        raw_line, = plt.plot(data["timesteps"], mean_reward, alpha=0.25)  # noisy raw curve, faint
        plt.plot(data["timesteps"], smoothed, color=raw_line.get_color(), label=algo.upper())
    plt.axhline(SOLVED_SCORE, color="gray", linestyle="--", label="solved (200)")
    plt.xlabel("Environment steps")
    plt.ylabel("Mean evaluation reward (5 episodes)")
    plt.title("LunarLander-v3: DQN vs PPO")
    plt.legend()
    plt.savefig("logs/reward_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    rows = []
    for algo in ALGOS:
        model = load_best_model(algo)
        print(f"Scoring {algo.upper()}...")
        rows.append({"algo": algo.upper(), **score_model(model)})
        record_videos(model, algo)

    table = pd.DataFrame(rows).set_index("algo").round(1)
    table.to_csv("logs/comparison.csv")  # read by app.py
    print(table)
    plot_reward_curves()
    print("Saved: logs/comparison.csv, logs/reward_curves.png, videos/")


if __name__ == "__main__":
    main()
"""Gradio demo: DQN vs PPO on LunarLander-v3, using results pre-computed by evaluate.py."""
from pathlib import Path

import gradio as gr
import pandas as pd

BASE_DIR = Path(__file__).parent  # absolute paths, so the app works from any launch directory
LOGS_DIR = BASE_DIR / "logs"
VIDEOS_DIR = BASE_DIR / "videos"

results = pd.read_csv(LOGS_DIR / "comparison.csv")

with gr.Blocks(title="LunarLander: DQN vs PPO") as demo:
    gr.Markdown(
        "# LunarLander-v3: DQN vs PPO\n"
        "One training run per algorithm (500,000 steps, seed 42). "
        "Scores are from 100 fresh episodes with a deterministic policy; "
        "videos show a single episode each."
    )
    gr.Dataframe(value=results, label="Comparison (reward of 200+ counts as solved)")
    gr.Image(value=str(LOGS_DIR / "reward_curves.png"), label="Reward during training", interactive=False)
    with gr.Row():
        gr.Video(value=str(VIDEOS_DIR / "dqn-episode-0.mp4"), label="DQN rollout", interactive=False)
        gr.Video(value=str(VIDEOS_DIR / "ppo-episode-0.mp4"), label="PPO rollout", interactive=False)

if __name__ == "__main__":
    # The docs say Gradio only serves local files that are in allowed_paths or its cache
    demo.launch(allowed_paths=[str(VIDEOS_DIR), str(LOGS_DIR)])
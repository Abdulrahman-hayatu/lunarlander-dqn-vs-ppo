"""Train DQN or PPO on LunarLander-v3 with Stable-Baselines3."""
import argparse

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

ENV_ID = "LunarLander-v3"
SEED = 42
N_EVAL_EPISODES = 5  # per evaluation; few, to keep evaluation cheap

# Hyperparameters from the RL Zoo model cards for LunarLander-v2; not retuned for v3
DQN_PARAMS = dict(
    learning_rate=6.3e-4, buffer_size=50_000, learning_starts=0, batch_size=128,
    gamma=0.99, train_freq=4, gradient_steps=-1, target_update_interval=250,
    exploration_fraction=0.12, exploration_final_eps=0.1,
    policy_kwargs=dict(net_arch=[256, 256]),
)
PPO_PARAMS = dict(n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01)
N_ENVS = {"dqn": 1, "ppo": 16}  # the Zoo trains PPO on 16 parallel environments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--timesteps", type=int, default=500_000, help="total training steps")
    parser.add_argument("--eval-every", type=int, default=10_000, help="evaluate every N total env steps")
    args = parser.parse_args()

    n_envs = N_ENVS[args.algo]
    train_env = make_vec_env(ENV_ID, n_envs=n_envs, seed=SEED)
    eval_env = make_vec_env(ENV_ID, n_envs=1, seed=SEED + 1)  # separate env for evaluation

    # eval_freq counts callback calls (one per vectorized step), so divide by n_envs
    # to get the same schedule in total env steps for both algorithms
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=max(args.eval_every // n_envs, 1),
        deterministic=True,
        best_model_save_path=f"models/{args.algo}",
        log_path=f"logs/{args.algo}",
    )

    if args.algo == "dqn":
        model = DQN("MlpPolicy", train_env, seed=SEED, verbose=1, **DQN_PARAMS)
    else:
        model = PPO("MlpPolicy", train_env, seed=SEED, verbose=1, **PPO_PARAMS)

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    print(f"Done. Best model: models/{args.algo}/best_model.zip | eval log: logs/{args.algo}/evaluations.npz")


if __name__ == "__main__":
    main()
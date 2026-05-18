"""
PPO training for Snake using Stable-Baselines3.
Uses the flat 11-feature observation space (same as Q-Learning and DQN-MLP)
so all three algorithms are compared on equal footing.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import sys
import os

# Allow running from repo root or from ppo/ subfolder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from game_env.snake_env import SnakeEnv

# ── paths ──────────────────────────────────────────────────────────────────
PPO_DIR = Path(__file__).resolve().parent
MODEL_PATH = PPO_DIR / "ppo_snake_model"
REWARDS_PLOT = PPO_DIR / "ppo_rewards.png"
SCORES_PLOT  = PPO_DIR / "ppo_scores.png"


# ── callback: records score (food eaten) each episode ──────────────────────
class ScoreCallback(BaseCallback):
    """Logs the game score (food eaten) and episode reward after each episode."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_scores   = []
        self.episode_rewards  = []
        self._current_reward  = 0.0

    def _on_step(self) -> bool:
        # SB3 stores per-step info in self.locals["infos"]
        self._current_reward += self.locals["rewards"][0]
        info = self.locals["infos"][0]
        if self.locals["dones"][0]:
            self.episode_scores.append(info.get("score", 0))
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


# ── training ───────────────────────────────────────────────────────────────
def train(
    timesteps: int = 500_000,
    grid_size: int = 8,
    living_cost: bool = False,
    seed: int = 42,
    model_name: str = "ppo_snake_model",
    verbose: int = 1,
):
    """
    Train PPO on Snake.

    Parameters
    ----------
    timesteps   : total environment steps
    grid_size   : side length of the square grid
    living_cost : whether to use dense reward shaping
    seed        : random seed for reproducibility
    model_name  : filename (no extension) for the saved model
    verbose     : 0 = silent, 1 = progress bar
    """
    print(f"\n{'='*50}")
    print(f"Training PPO  |  grid={grid_size}x{grid_size}  |  steps={timesteps:,}  |  seed={seed}")
    print(f"{'='*50}\n")

    env = SnakeEnv(obs="flat", living_cost=living_cost,
                   width=grid_size, height=grid_size)
    env = Monitor(env)          # wraps env to auto-log episode stats

    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        verbose         = verbose,
        seed            = seed,
        # ── architecture ──────────────────────────────────────
        policy_kwargs   = dict(net_arch=[128, 128]),
        # ── key PPO hyper-parameters ──────────────────────────
        learning_rate   = 3e-4,
        n_steps         = 2048,   # steps collected per update
        batch_size      = 64,
        n_epochs        = 10,     # gradient steps per update
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,   # entropy bonus → encourages exploration
    )

    callback = ScoreCallback()
    model.learn(total_timesteps=timesteps, callback=callback)

    # ── save model ────────────────────────────────────────────
    save_path = PPO_DIR / model_name
    model.save(str(save_path))
    print(f"\nModel saved → {save_path}.zip")

    # ── save plots ────────────────────────────────────────────
    _plot_training(callback.episode_rewards, callback.episode_scores,
                   grid_size, timesteps)

    env.close()
    return model, callback.episode_scores, callback.episode_rewards


def _plot_training(rewards, scores, grid_size, timesteps):
    """Save learning-curve plots."""
    def running_mean(x, N=50):
        if len(x) < N:
            return np.array(x)
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PPO Training — {grid_size}×{grid_size} grid, {timesteps:,} steps")

    axes[0].plot(running_mean(rewards, 50), color="steelblue")
    axes[0].set_title("Episode Reward (running mean, window=50)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative reward")
    axes[0].grid(True)

    axes[1].plot(running_mean(scores, 50), color="darkorange")
    axes[1].set_title("Score / Food eaten (running mean, window=50)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Score")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(REWARDS_PLOT, dpi=150)
    plt.close()
    print(f"Plots saved → {REWARDS_PLOT}")


# ── evaluation ─────────────────────────────────────────────────────────────
def evaluate(
    model_name: str = "ppo_snake_model",
    episodes:   int = 100,
    grid_size:  int = 8,
    living_cost: bool = False,
    render: bool = False,
    seed: int = 42,
) -> dict:
    """
    Evaluate a saved PPO model.

    Returns a dict with mean/std/max score and mean survival steps,
    so it can be directly compared with Q-Learning and DQN results.
    """
    model_path = PPO_DIR / model_name
    env = SnakeEnv(
        obs="flat",
        living_cost=living_cost,
        width=grid_size,
        height=grid_size,
        render_mode="human" if render else None,
    )
    model = PPO.load(str(model_path), env=env)

    scores, survival_steps = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
        scores.append(info.get("score", 0))
        survival_steps.append(steps)

    env.close()

    results = {
        "algorithm":      "PPO",
        "grid_size":      grid_size,
        "episodes":       episodes,
        "mean_score":     float(np.mean(scores)),
        "std_score":      float(np.std(scores)),
        "max_score":      int(np.max(scores)),
        "mean_survival":  float(np.mean(survival_steps)),
    }
    print(f"\nPPO evaluation ({episodes} episodes, {grid_size}×{grid_size})")
    print(f"  Mean score  : {results['mean_score']:.2f} ± {results['std_score']:.2f}")
    print(f"  Max score   : {results['max_score']}")
    print(f"  Mean steps  : {results['mean_survival']:.1f}")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO Snake — train or evaluate")
    parser.add_argument("mode", choices=["train", "eval", "play"],
                        nargs="?", default="train")
    parser.add_argument("--timesteps",   type=int,   default=500_000)
    parser.add_argument("--grid",        type=int,   default=8)
    parser.add_argument("--living-cost", action="store_true")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--episodes",    type=int,   default=100)
    parser.add_argument("--model",       type=str,   default="ppo_snake_model")
    args = parser.parse_args()

    if args.mode == "train":
        train(
            timesteps   = args.timesteps,
            grid_size   = args.grid,
            living_cost = args.living_cost,
            seed        = args.seed,
            model_name  = args.model,
        )
    elif args.mode == "eval":
        evaluate(
            model_name  = args.model,
            episodes    = args.episodes,
            grid_size   = args.grid,
            living_cost = args.living_cost,
            seed        = args.seed,
        )
    elif args.mode == "play":
        evaluate(
            model_name  = args.model,
            episodes    = args.episodes,
            grid_size   = args.grid,
            living_cost = args.living_cost,
            render      = True,
            seed        = args.seed,
        )

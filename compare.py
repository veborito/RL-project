"""
compare.py — Unified evaluation and comparison of all RL algorithms.

Usage:
    python compare.py --mode eval_all --grid 8 --episodes 200
    python compare.py --mode plot_comparison
    python compare.py --mode ablation_grid      # vary grid size
    python compare.py --mode ablation_reward    # vary reward function

This script provides:
  1. A common evaluate() interface for Q-Learning, DQN-MLP, DQN-CNN, PPO
  2. Side-by-side bar/curve plots
  3. Ablation studies (grid size, reward shaping)
  4. Results saved to JSON for reproducibility
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from game_env.snake_env import SnakeEnv

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Algorithm evaluators
# ══════════════════════════════════════════════════════════════════════════════

def eval_random(episodes=200, grid_size=8, seed=42) -> dict:
    """Baseline: uniformly random actions."""
    env = SnakeEnv(obs="flat", width=grid_size, height=grid_size)
    rng = np.random.default_rng(seed)
    scores, survivals = [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done, steps = False, 0
        while not done:
            _, _, terminated, truncated, info = env.step(rng.integers(0, 4))
            done = terminated or truncated
            steps += 1
        scores.append(info.get("score", 0))
        survivals.append(steps)
    env.close()
    return _make_result("Random", grid_size, episodes, scores, survivals)


def eval_qlearning(model_name="q_learning_model_100k", episodes=200,
                   grid_size=8, seed=42) -> dict:
    """Evaluate saved Q-table."""
    model_path = ROOT / "q_learning" / (model_name + ".pkl")
    if not model_path.exists():
        print(f"[WARN] Q-Learning model not found: {model_path}")
        return {}

    with open(model_path, "rb") as f:
        q = pickle.load(f)

    env = SnakeEnv(obs="flat", width=grid_size, height=grid_size)
    scores, survivals = [], []
    for ep in range(episodes):
        state, _ = env.reset_bin(seed=seed + ep)
        done, steps = False, 0
        while not done:
            action = int(np.argmax(q[state]))
            state, _, terminated, truncated, info = env.step_bin(action)
            done = terminated or truncated
            steps += 1
        scores.append(info.get("score", 0))
        survivals.append(steps)
    env.close()
    return _make_result("Q-Learning", grid_size, episodes, scores, survivals)


def eval_dqn(model_file="snake_dql.pt", episodes=200,
             grid_size=8, seed=42) -> dict:
    """Evaluate saved DQN-MLP weights."""
    import torch, torch.nn as nn

    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu')
    model_path = ROOT / model_file
    if not model_path.exists():
        print(f"[WARN] DQN model not found: {model_path}")
        return {}

    # inline QNetwork (same architecture as dqn.ipynb)
    class QNetwork(nn.Module):
      def __init__(self, n_states, n_actions, hidden_dim):
          super(QNetwork, self).__init__()
          self.linear1 = nn.Linear(n_states, hidden_dim)
          self.linear2 = nn.Linear(hidden_dim, hidden_dim)
          self.linear3 = nn.Linear(hidden_dim, n_actions)

      def forward(self, state):
          x = F.relu(self.linear1(state))
          x = F.relu(self.linear2(x))
          return self.linear3(x)
      
    net = QNetwork(11, 4, 128).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    env = SnakeEnv(obs="flat", width=grid_size, height=grid_size)
    scores, survivals = [], []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done, steps = False, 0
        while not done:
            with torch.no_grad():
                q = net(torch.tensor(state, dtype=torch.float32).to(device))
                action = int(torch.argmax(q).item())
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        scores.append(info.get("score", 0))
        survivals.append(steps)
    env.close()
    return _make_result("DQN-MLP", grid_size, episodes, scores, survivals)


# def eval_dqn_cnn(model_file="dqn/dqn_cnn_model.pt", episodes=200,
#                  grid_size=8, seed=42) -> dict:
#     """Evaluate saved DQN-CNN weights."""
#     import torch, torch.nn as nn

#     device = torch.device("cpu")
#     model_path = ROOT / model_file
#     if not model_path.exists():
#         print(f"[WARN] DQN-CNN model not found: {model_path}")
#         return {}

#     sys.path.insert(0, str(ROOT / "dqn"))
#     from dqn.dqn_cnn import QNetwork, state_to_dqn_input

#     net = QNetworkCNN(grid_size, grid_size, out_actions=4).to(device)
#     net.load_state_dict(torch.load(model_path, map_location=device))
#     net.eval()

#     env = SnakeEnv(obs="game", width=grid_size, height=grid_size)
#     scores, survivals = [], []
#     for ep in range(episodes):
#         state, _ = env.reset(seed=seed + ep)
#         done, steps = False, 0
#         while not done:
#             with torch.no_grad():
#                 q      = net(state_to_dqn_input(state))
#                 action = int(torch.argmax(q).item())
#             state, _, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             steps += 1
#         scores.append(info.get("score", 0))
#         survivals.append(steps)
#     env.close()
#     return _make_result("DQN-CNN", grid_size, episodes, scores, survivals)


def eval_ppo(model_name="ppo_snake_model", episodes=200,
             grid_size=8, living_cost=False, seed=42) -> dict:
    """Evaluate saved PPO model."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[WARN] stable-baselines3 not installed. Skipping PPO eval.")
        return {}

    model_path = ROOT / "ppo" / model_name
    if not (str(model_path) + ".zip").__class__(str(model_path) + ".zip").startswith("/"):
        pass  # just a path check
    if not Path(str(model_path) + ".zip").exists():
        print(f"[WARN] PPO model not found: {model_path}.zip")
        return {}

    env = SnakeEnv(obs="flat", living_cost=living_cost,
                   width=grid_size, height=grid_size)
    model = PPO.load(str(model_path), env=env)

    scores, survivals = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, steps = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
        scores.append(info.get("score", 0))
        survivals.append(steps)
    env.close()
    return _make_result("PPO", grid_size, episodes, scores, survivals)


def _make_result(algo, grid_size, episodes, scores, survivals) -> dict:
    scores    = np.array(scores)
    survivals = np.array(survivals)
    r = {
        "algorithm":     algo,
        "grid_size":     grid_size,
        "episodes":      episodes,
        "mean_score":    float(scores.mean()),
        "std_score":     float(scores.std()),
        "max_score":     int(scores.max()),
        "mean_survival": float(survivals.mean()),
        "std_survival":  float(survivals.std()),
        "scores":        scores.tolist(),
        "survivals":     survivals.tolist(),
    }
    print(f"  [{algo:12s}] score={r['mean_score']:.2f}±{r['std_score']:.2f}"
          f"  max={r['max_score']}  survival={r['mean_survival']:.1f}")
    return r


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Plots
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Random":    "#999999",
    "Q-Learning":"#e41a1c",
    "DQN-MLP":   "#377eb8",
    "DQN-CNN":   "#ff7f00",
    "PPO":       "#4daf4a",
}


def plot_comparison(results: list[dict], tag: str = ""):
    """Bar chart comparing mean score and mean survival across algorithms."""
    algos     = [r["algorithm"]     for r in results]
    means     = [r["mean_score"]    for r in results]
    stds      = [r["std_score"]     for r in results]
    survivals = [r["mean_survival"] for r in results]
    colors    = [COLORS.get(a, "#888") for a in algos]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    title_suffix = f" — {tag}" if tag else ""
    fig.suptitle(f"Algorithm Comparison{title_suffix}", fontsize=14, fontweight="bold")

    # Score bars
    bars = axes[0].bar(algos, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    axes[0].set_title("Mean Score (food eaten)")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, max(means) * 1.3 + 0.5)
    for bar, m in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Survival bars
    bars2 = axes[1].bar(algos, survivals, color=colors, alpha=0.85)
    axes[1].set_title("Mean Survival (steps/episode)")
    axes[1].set_ylabel("Steps")
    axes[1].set_ylim(0, max(survivals) * 1.3 + 1)
    for bar, s in zip(bars2, survivals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{s:.0f}", ha="center", va="bottom", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = RESULTS_DIR / f"comparison{'_' + tag if tag else ''}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved → {fname}")


def plot_score_distributions(results: list[dict], tag: str = ""):
    """Box plots of score distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data   = [r["scores"]    for r in results]
    labels = [r["algorithm"] for r in results]
    colors = [COLORS.get(a, "#888") for a in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(f"Score Distribution{' — ' + tag if tag else ''}")
    ax.set_ylabel("Score (food eaten)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = RESULTS_DIR / f"distributions{'_' + tag if tag else ''}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved → {fname}")


def plot_ablation_grid(results_by_grid: dict):
    """
    results_by_grid: { grid_size: [list of result dicts] }
    Plots mean score vs grid size for each algorithm.
    """
    algos = list({r["algorithm"] for rlist in results_by_grid.values() for r in rlist})
    grids = sorted(results_by_grid.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in algos:
        means, stds = [], []
        for g in grids:
            rlist = results_by_grid[g]
            match = [r for r in rlist if r["algorithm"] == algo]
            if match:
                means.append(match[0]["mean_score"])
                stds.append(match[0]["std_score"])
            else:
                means.append(None); stds.append(None)
        valid = [(g, m, s) for g, m, s in zip(grids, means, stds) if m is not None]
        if valid:
            gs, ms, ss = zip(*valid)
            ax.errorbar(gs, ms, yerr=ss, label=algo, marker="o",
                        color=COLORS.get(algo, "#888"), linewidth=2, capsize=4)

    ax.set_title("Mean Score vs Grid Size")
    ax.set_xlabel("Grid size (N×N)")
    ax.set_ylabel("Mean Score")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = RESULTS_DIR / "ablation_grid.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Experiment runners
# ══════════════════════════════════════════════════════════════════════════════

def run_eval_all(grid_size=8, episodes=200, seed=42):
    """Evaluate all available trained models on a single grid size."""
    print(f"\n{'='*55}")
    print(f"Evaluating all algorithms  |  grid={grid_size}x{grid_size}  |  episodes={episodes}")
    print(f"{'='*55}")

    results = []
    for fn, kwargs in [
        (eval_random,   dict(grid_size=grid_size, episodes=episodes, seed=seed)),
        (eval_qlearning,dict(grid_size=grid_size, episodes=episodes, seed=seed)),
        (eval_dqn,      dict(grid_size=grid_size, episodes=episodes, seed=seed)),
        # (eval_dqn_cnn,  dict(grid_size=grid_size, episodes=episodes, seed=seed)),
        (eval_ppo,      dict(grid_size=grid_size, episodes=episodes, seed=seed)),
    ]:
        r = fn(**kwargs)
        if r:
            results.append(r)

    tag = f"grid{grid_size}"
    _save_results(results, f"eval_all_{tag}.json")
    plot_comparison(results, tag)
    plot_score_distributions(results, tag)
    return results


def run_ablation_grid(grid_sizes=(5, 8, 10), episodes=100, seed=42):
    """Evaluate all algorithms across multiple grid sizes."""
    results_by_grid = {}
    for g in grid_sizes:
        results_by_grid[g] = run_eval_all(grid_size=g, episodes=episodes, seed=seed)

    _save_results(results_by_grid, "ablation_grid.json")
    plot_ablation_grid(results_by_grid)
    return results_by_grid


def _save_results(data, filename: str):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eval_all", "ablation_grid"],
                   default="eval_all")
    p.add_argument("--grid",     type=int,   default=8)
    p.add_argument("--episodes", type=int,   default=200)
    p.add_argument("--seed",     type=int,   default=42)
    args = p.parse_args()

    if args.mode == "eval_all":
        run_eval_all(grid_size=args.grid, episodes=args.episodes, seed=args.seed)
    elif args.mode == "ablation_grid":
        run_ablation_grid(grid_sizes=(5, 8, 10), episodes=args.episodes, seed=args.seed)

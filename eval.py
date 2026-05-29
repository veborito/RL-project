"""
AI-generated, we did it with the help of Claude.

Final evaluation script for the Snake RL project

Produces all results needed for the final report:
  1. Algorithm comparison  : sparse vs dense, 8×8 grid, 5 seeds
  2. Grid size ablation    : 8×8, 10×10, 15×15, 20×20 (flat-obs models)
  3. Obstacles ablation    : 0 vs 4 obstacles, 8×8 grid

Usage:
    python eval.py --mode all          # run everything
    python eval.py --mode comparison   # sparse vs dense comparison
    python eval.py --mode grid         # grid size ablation
    python eval.py --mode obstacles    # obstacles ablation
    python eval.py --mode plots        # regenerate plots from saved JSON

All results saved to results/final/ as JSON + PNG.
"""

import os, sys, json, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "final"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 999]
N_EVAL_EPISODES = 200
MAX_STEPS = 500

# colours matching training plots
C = {
    "Random":       "#999999",
    "Q-Learning":   "#E55C5C",
    "DQN-MLP":      "#5C8FE5",
    "PPO":          "#3FB950",
}


# Helpers

def bootstrap_ci(data, n_boot=1000, ci=0.95):
    """Return (mean, lower, upper) with bootstrapped CI."""
    data = np.array(data)
    means = [np.mean(np.random.choice(data, size=len(data), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(data)), float(lo), float(hi)


def make_env(grid=8, living_cost=False, obstacles=False, num_obstacles=4):
    """Instantiate the correct environment."""
    if obstacles:
        from game_env.snake_env import SnakeEnv
        from game_env.snake_w_obstacles import SnakeGameObstacles
        import gymnasium as gym
        from gymnasium import spaces

        class SnakeObsEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.game = SnakeGameObstacles(
                    obs="flat", living_cost=living_cost,
                    width=grid, height=grid,
                    num_obstacles=num_obstacles,
                )
                self.action_space = spaces.Discrete(4)
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(11,), dtype=np.float32)

            def reset(self, seed=None, options=None):
                if seed is not None:
                    np.random.seed(seed)
                obs = self.game.reset()
                return obs.astype(np.float32), {"score": self.game.score}

            def step(self, action):
                obs, r, term, trunc, info = self.game.take_action(int(action))
                return obs.astype(np.float32), r, term, trunc, info

            def reset_bin(self, seed=None):
                obs, info = self.reset(seed=seed)
                bin_str = np.array2string(obs.astype(int)).strip("[]").replace(" ", "")
                return int(bin_str, 2), info

            def step_bin(self, action):
                obs, r, term, trunc, info = self.step(action)
                bin_str = np.array2string(obs.astype(int)).strip("[]").replace(" ", "")
                return int(bin_str, 2), r, term, trunc, info

            def close(self): self.game.close()

        return SnakeObsEnv()
    else:
        from game_env.snake_env import SnakeEnv
        return SnakeEnv(width=grid, height=grid, living_cost=living_cost)



# Evaluators (one per algorithm family)

def eval_random(grid=8, obstacles=False, seeds=SEEDS, n_ep=N_EVAL_EPISODES):
    scores, survivals = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env = make_env(grid=grid, obstacles=obstacles)
        for ep in range(n_ep):
            env.reset(seed=seed + ep)
            done, steps = False, 0
            while not done and steps < MAX_STEPS:
                _, _, terminated, truncated, info = env.step(int(rng.integers(0, 4)))
                done = terminated or truncated
                steps += 1
            scores.append(info.get("score", 0))
            survivals.append(steps)
        env.close()
    return _result("Random", grid, scores, survivals, obstacles)


def eval_qlearning(model_file, label="Q-Learning", grid=8,
                   obstacles=False, seeds=SEEDS, n_ep=N_EVAL_EPISODES):
    model_path = ROOT / "q_learning" / model_file
    if not model_path.exists():
        print(f"  [SKIP] {model_path} not found")
        return None
    with open(model_path, "rb") as f:
        q = pickle.load(f)

    scores, survivals = [], []
    for seed in seeds:
        env = make_env(grid=grid, obstacles=obstacles)
        for ep in range(n_ep):
            state, _ = env.reset_bin(seed=seed + ep)
            done, steps = False, 0
            while not done and steps < MAX_STEPS:
                action = int(np.argmax(q[state]))
                state, _, terminated, truncated, info = env.step_bin(action)
                done = terminated or truncated
                steps += 1
            scores.append(info.get("score", 0))
            survivals.append(steps)
        env.close()
    return _result(label, grid, scores, survivals, obstacles)


def eval_dqn(model_file, label="DQN-MLP", grid=8,
             obstacles=False, seeds=SEEDS, n_ep=N_EVAL_EPISODES):
    import torch, torch.nn as nn, torch.nn.functional as F

    model_path = ROOT / model_file
    if not model_path.exists():
        print(f"  [SKIP] {model_path} not found")
        return None

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(11, 128)
            self.linear2 = nn.Linear(128, 128)
            self.linear3 = nn.Linear(128, 4)
        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            return self.linear3(x)

    net = QNet()
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    scores, survivals = [], []
    for seed in seeds:
        env = make_env(grid=grid, obstacles=obstacles)
        for ep in range(n_ep):
            state, _ = env.reset(seed=seed + ep)
            done, steps = False, 0
            while not done and steps < MAX_STEPS:
                with torch.no_grad():
                    q = net(torch.tensor(state, dtype=torch.float32))
                    action = int(torch.argmax(q).item())
                state, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            scores.append(info.get("score", 0))
            survivals.append(steps)
        env.close()
    return _result(label, grid, scores, survivals, obstacles)


def eval_ppo(model_file, label="PPO", grid=8,
             living_cost=False, obstacles=False, seeds=SEEDS, n_ep=N_EVAL_EPISODES):
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("  [SKIP] stable-baselines3 not installed")
        return None

    model_path = ROOT / "ppo" / model_file
    if not Path(str(model_path) + ".zip").exists() and not model_path.exists():
        print(f"  [SKIP] {model_path} not found")
        return None

    env = make_env(grid=grid, living_cost=living_cost, obstacles=obstacles)
    model = PPO.load(str(model_path), env=env)

    scores, survivals = [], []
    for seed in seeds:
        for ep in range(n_ep):
            obs, _ = env.reset(seed=seed + ep)
            done, steps = False, 0
            while not done and steps < MAX_STEPS:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                steps += 1
            scores.append(info.get("score", 0))
            survivals.append(steps)
    env.close()
    return _result(label, grid, scores, survivals, obstacles)


def _result(algo, grid, scores, survivals, obstacles):
    scores, survivals = np.array(scores), np.array(survivals)
    s_mean, s_lo, s_hi = bootstrap_ci(scores)
    v_mean, v_lo, v_hi = bootstrap_ci(survivals)
    r = {
        "algorithm": algo,
        "grid_size": grid,
        "obstacles": obstacles,
        "n_eval": len(scores),
        "mean_score": s_mean, "ci_lo_score": s_lo, "ci_hi_score": s_hi,
        "std_score": float(scores.std()),
        "max_score": int(scores.max()),
        "mean_survival": v_mean, "ci_lo_survival": v_lo, "ci_hi_survival": v_hi,
        "scores": scores.tolist(),
        "survivals": survivals.tolist(),
    }
    print(f"  [{algo:20s}] score={s_mean:.2f} [{s_lo:.2f},{s_hi:.2f}]  "
          f"max={r['max_score']}  survival={v_mean:.1f}")
    return r


# Experiment runners

def run_comparison():
    """Sparse vs Dense comparison on 8×8 grid."""
    print("\n" + "="*60)
    print("EXPERIMENT 1 — Sparse vs Dense, 8×8, no obstacles")
    print("="*60)

    sparse, dense = [], []

    # Random (same for both)
    r = eval_random()
    r["reward"] = "sparse"; sparse.append(r)
    r2 = dict(r); r2["reward"] = "dense"; dense.append(r2)

    # Q-Learning
    r = eval_qlearning("q_learning_model_10k_sparse.pkl", "Q-Learning")
    if r: r["reward"] = "sparse"; sparse.append(r)
    r = eval_qlearning("q_learning_model_10k_dense.pkl", "Q-Learning")
    if r: r["reward"] = "dense"; dense.append(r)

    # DQN-MLP sparse (10k only)
    r = eval_dqn("snake_dql_sparse_10k.pt", "DQN-MLP")
    if r: r["reward"] = "sparse"; sparse.append(r)

    # DQN-MLP dense (10k only)
    r = eval_dqn("snake_dql_dense_10k.pt", "DQN-MLP")
    if r: r["reward"] = "dense"; dense.append(r)

    # PPO sparse
    r = eval_ppo("ppo_snake_model_sparse", "PPO", living_cost=False)
    if r: r["reward"] = "sparse"; sparse.append(r)

    # PPO dense
    r = eval_ppo("ppo_snake_model_dense", "PPO", living_cost=True)
    if r: r["reward"] = "dense"; dense.append(r)

    data = {"sparse": sparse, "dense": dense}
    _save(data, "comparison.json")
    plot_comparison(sparse, dense)
    return data


def run_grid_ablation():
    """Grid size ablation using flat-obs models (trained on 8×8)."""
    print("\n" + "="*60)
    print("EXPERIMENT 2 — Grid size ablation")
    print("="*60)
    results = {}
    for grid in [8, 10, 15, 20]:
        print(f"\n  Grid {grid}×{grid}")
        row = []
        row.append(eval_random(grid=grid))
        r = eval_qlearning("q_learning_model_10k_sparse.pkl", "Q-Learning", grid=grid)
        if r: row.append(r)
        r = eval_dqn("snake_dql_sparse_10k.pt", "DQN-MLP", grid=grid)
        if r: row.append(r)
        r = eval_ppo("ppo_snake_model_sparse", "PPO", grid=grid, living_cost=False)
        if r: row.append(r)
        results[grid] = row

    _save(results, "grid_ablation.json")
    plot_grid_ablation(results)
    return results


def run_obstacles_ablation():
    """Obstacles ablation: 0 vs 4 random obstacles, 8×8."""
    print("\n" + "="*60)
    print("EXPERIMENT 3 — Obstacles ablation (0 vs 4), 8×8")
    print("="*60)

    no_obs, with_obs = [], []

    for obstacles, lst in [(False, no_obs), (True, with_obs)]:
        tag = "4 obstacles" if obstacles else "no obstacles"
        print(f"\n  {tag}")
        lst.append(eval_random(obstacles=obstacles))
        r = eval_qlearning("q_learning_model_10k_sparse.pkl", "Q-Learning", obstacles=obstacles)
        if r: lst.append(r)
        r = eval_dqn("snake_dql_sparse_10k.pt", "DQN-MLP", obstacles=obstacles)
        if r: lst.append(r)
        r = eval_ppo("ppo_snake_model_sparse", "PPO", obstacles=obstacles, living_cost=False)
        if r: lst.append(r)

    data = {"no_obstacles": no_obs, "with_obstacles": with_obs}
    _save(data, "obstacles_ablation.json")
    plot_obstacles(no_obs, with_obs)
    return data


# Plots

def _bar(ax, results, metric, ylabel, title, with_ci=True):
    algos = [r["algorithm"] for r in results]
    means = [r[f"mean_{metric}"] for r in results]
    lo    = [r[f"mean_{metric}"] - r[f"ci_lo_{metric}"] for r in results]
    hi    = [r[f"ci_hi_{metric}"] - r[f"mean_{metric}"] for r in results]
    colors = [C.get(a.split("-")[0] if "-" in a else a, "#888") for a in algos]
    bars = ax.bar(algos, means, color=colors, alpha=0.85,
                  yerr=[lo, hi] if with_ci else None, capsize=5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(hi)*0.05,
                f"{m:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)


def plot_comparison(sparse, dense):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sparse vs Dense Reward — 8×8 Grid", fontsize=14, fontweight="bold")

    for col, (results, tag) in enumerate([(sparse, "Sparse"), (dense, "Dense")]):
        _bar(axes[0, col], results, "score",    "Mean Score",    f"{tag} — Score")
        _bar(axes[1, col], results, "survival", "Mean Survival", f"{tag} — Survival")

    plt.tight_layout()
    p = RESULTS_DIR / "comparison_sparse_vs_dense.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved → {p}")

    # also grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sparse vs Dense", fontsize=13, fontweight="bold")
    for ax_idx, metric in enumerate(["score", "survival"]):
        # align by algo name
        algo_names = list(dict.fromkeys(
            [r["algorithm"] for r in sparse] + [r["algorithm"] for r in dense]))
        s_map = {r["algorithm"]: r for r in sparse}
        d_map = {r["algorithm"]: r for r in dense}
        x = np.arange(len(algo_names)); w = 0.35
        for i, algo in enumerate(algo_names):
            color = C.get(algo.split("-")[0] if "-" in algo else algo, "#888")
            if algo in s_map:
                r = s_map[algo]
                axes[ax_idx].bar(x[i] - w/2, r[f"mean_{metric}"],
                    width=w, color=color, alpha=0.9, label="Sparse" if i==0 else "")
            if algo in d_map:
                r = d_map[algo]
                axes[ax_idx].bar(x[i] + w/2, r[f"mean_{metric}"],
                    width=w, color=color, alpha=0.5, hatch="//",
                    label="Dense" if i==0 else "")
        axes[ax_idx].set_xticks(x); axes[ax_idx].set_xticklabels(algo_names, rotation=15)
        axes[ax_idx].set_ylabel("Mean Score" if metric == "score" else "Mean Survival (steps)")
        axes[ax_idx].grid(axis="y", alpha=0.3)
        axes[ax_idx].legend(["Sparse", "Dense"])
    plt.tight_layout()
    p2 = RESULTS_DIR / "comparison_grouped.png"
    plt.savefig(p2, dpi=150); plt.close()
    print(f"  Saved → {p2}")


def plot_grid_ablation(results):
    grids = sorted(results.keys())
    algos = list(dict.fromkeys(
        r["algorithm"] for g in grids for r in results[g]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Grid Size Ablation", fontsize=13, fontweight="bold")

    for ax_idx, metric in enumerate(["score", "survival"]):
        for algo in algos:
            xs, ys, los, his = [], [], [], []
            for g in grids:
                match = [r for r in results[g] if r["algorithm"] == algo]
                if match:
                    r = match[0]
                    xs.append(g); ys.append(r[f"mean_{metric}"])
                    los.append(r[f"mean_{metric}"] - r[f"ci_lo_{metric}"])
                    his.append(r[f"ci_hi_{metric}"] - r[f"mean_{metric}"])
            color = C.get(algo.split("-")[0] if "-" in algo else algo, "#888")
            axes[ax_idx].errorbar(xs, ys, yerr=[los, his], label=algo,
                marker="o", color=color, linewidth=2, capsize=4)
        axes[ax_idx].set_xlabel("Grid size (N×N)")
        axes[ax_idx].set_ylabel("Mean Score" if metric == "score" else "Mean Survival (steps)")
        axes[ax_idx].set_title("Score vs Grid Size" if metric == "score" else "Survival vs Grid Size")
        axes[ax_idx].legend(); axes[ax_idx].grid(alpha=0.3)
        axes[ax_idx].set_xticks(grids)
        axes[ax_idx].set_xticklabels([f"{g}×{g}" for g in grids])

    plt.tight_layout()
    p = RESULTS_DIR / "grid_ablation.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved → {p}")


def plot_obstacles(no_obs, with_obs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Obstacles Ablation — 0 vs 4 Obstacles (8×8)", fontsize=13, fontweight="bold")

    for ax_idx, metric in enumerate(["score", "survival"]):
        algos = [r["algorithm"] for r in no_obs]
        x = np.arange(len(algos)); w = 0.35
        no_means  = [r[f"mean_{metric}"] for r in no_obs]
        no_hi     = [r[f"ci_hi_{metric}"] - r[f"mean_{metric}"] for r in no_obs]
        no_lo     = [r[f"mean_{metric}"] - r[f"ci_lo_{metric}"] for r in no_obs]
        obs_map   = {r["algorithm"]: r for r in with_obs}

        colors = [C.get(a.split("-")[0] if "-" in a else a, "#888") for a in algos]
        axes[ax_idx].bar(x - w/2, no_means, width=w, color=colors, alpha=0.9,
                         yerr=[no_lo, no_hi], capsize=4, label="No obstacles")

        obs_means, obs_lo, obs_hi = [], [], []
        for algo in algos:
            if algo in obs_map:
                r = obs_map[algo]
                obs_means.append(r[f"mean_{metric}"])
                obs_lo.append(r[f"mean_{metric}"] - r[f"ci_lo_{metric}"])
                obs_hi.append(r[f"ci_hi_{metric}"] - r[f"mean_{metric}"])
            else:
                obs_means.append(0); obs_lo.append(0); obs_hi.append(0)

        axes[ax_idx].bar(x + w/2, obs_means, width=w, color=colors, alpha=0.5,
                         hatch="//", yerr=[obs_lo, obs_hi], capsize=4, label="4 obstacles")

        axes[ax_idx].set_xticks(x); axes[ax_idx].set_xticklabels(algos, rotation=15)
        axes[ax_idx].set_ylabel("Mean Score" if metric == "score" else "Mean Survival (steps)")
        axes[ax_idx].set_title("Score" if metric == "score" else "Survival")
        axes[ax_idx].legend(); axes[ax_idx].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = RESULTS_DIR / "obstacles_ablation.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved → {p}")


def plot_score_distributions(results, tag=""):
    """Box plots for a list of result dicts."""
    fig, ax = plt.subplots(figsize=(10, 5))
    algos  = [r["algorithm"] for r in results]
    data   = [r["scores"]    for r in results]
    colors = [C.get(a.split("-")[0] if "-" in a else a, "#888") for a in algos]
    bp = ax.boxplot(data, tick_labels=algos, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title(f"Score Distribution{' — ' + tag if tag else ''}")
    ax.set_ylabel("Score (food eaten)"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = RESULTS_DIR / f"distribution{'_' + tag.replace(' ','_') if tag else ''}.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved → {fname}")


# Utils

def _save(data, fname):
    p = RESULTS_DIR / fname
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved → {p}")


def load_and_plot():
    """Regenerate plots from saved JSON files."""
    for fname, fn in [
        ("comparison.json",       lambda d: plot_comparison(d["sparse"], d["dense"])),
        ("grid_ablation.json",    lambda d: plot_grid_ablation({int(k): v for k,v in d.items()})),
        ("obstacles_ablation.json", lambda d: plot_obstacles(d["no_obstacles"], d["with_obstacles"])),
    ]:
        p = RESULTS_DIR / fname
        if p.exists():
            with open(p) as f: data = json.load(f)
            print(f"Plotting from {fname}...")
            fn(data)


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all","comparison","grid","obstacles","plots"],
                        default="all")
    parser.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--seeds",    type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    N_EVAL_EPISODES = args.episodes
    SEEDS = args.seeds

    if args.mode in ("all", "comparison"):
        run_comparison()
    if args.mode in ("all", "grid"):
        run_grid_ablation()
    if args.mode in ("all", "obstacles"):
        run_obstacles_ablation()
    if args.mode == "plots":
        load_and_plot()

    print("\nResults in results/final/")

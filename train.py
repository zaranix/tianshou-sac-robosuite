# train_sb3_sac_lift_repro_nominal_confounded.py
# Same as your working baseline, but:
#  - nominal spurious correlation (pos <-> color) like the paper
#  - append RGB (3 dims) to state so agent can learn the color correlation

import os
import json
import random
import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from success import SuccessInfoWrapper
from spurious_lift import NominalColorPosToStateWrapper  # <-- NEW


# -----------------------------
# Utilities
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_versions(path: str) -> None:
    import stable_baselines3 as sb3
    import gymnasium

    lines = [
        f"time: {datetime.datetime.now().isoformat()}",
        f"python: {os.sys.version.replace(os.linesep, ' ')}",
        f"torch: {torch.__version__}",
        f"stable_baselines3: {sb3.__version__}",
        f"robosuite: {suite.__version__}",
        f"gymnasium: {gymnasium.__version__}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Environment
# -----------------------------
def make_env(
    *,
    seed: Optional[int] = None,
    horizon: int = 300,
    control_freq: int = 20,
    reward_shaping: bool = True,
    spurious_seed: Optional[int] = None,  # <-- NEW
) -> Monitor:
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,  # state-based
        control_freq=control_freq,
        horizon=horizon,
        reward_shaping=reward_shaping,
    )

    env = GymWrapper(env)

    # NEW: nominal spurious correlation + add RGB to state
    if spurious_seed is None:
        spurious_seed = 0 if seed is None else int(seed)
    env = NominalColorPosToStateWrapper(env, seed=spurious_seed)

    env = SuccessInfoWrapper(env)
    env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def get_obs_act_dims(env) -> Tuple[int, int]:
    obs, info = env.reset()
    obs_dim = int(np.asarray(obs).shape[0])
    act_dim = int(np.prod(env.action_space.shape))
    return obs_dim, act_dim


# -----------------------------
# Evaluation callback (best by success)
# -----------------------------
class SuccessEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_episodes: int = 50, eval_freq: int = 50_000, save_dir: str = "checkpoints", verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = int(eval_episodes)
        self.eval_freq = int(eval_freq)
        self.save_dir = save_dir
        self.best_sr = -1.0
        ensure_dir(self.save_dir)

    def _evaluate(self) -> float:
        succ = 0.0
        for _ in range(self.eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            ep_succ = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_succ = max(ep_succ, float(info.get("success", 0.0)))
                done = bool(terminated) or bool(truncated)
            succ += ep_succ
        return float(succ / self.eval_episodes)

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and (self.num_timesteps % self.eval_freq == 0):
            sr = self._evaluate()
            try:
                self.logger.record("eval/success_rate", sr)
            except Exception:
                pass

            if self.verbose:
                print(f"\n[Eval] steps={self.num_timesteps} success_rate={sr:.3f}\n")

            if sr > self.best_sr:
                self.best_sr = sr
                best_path = os.path.join(self.save_dir, "best_by_success")
                self.model.save(best_path)
                if self.verbose:
                    print(f"✅ Saved best checkpoint: {best_path}.zip (sr={sr:.3f})")
        return True


def eval_success_rate(
    model: SAC,
    *,
    episodes: int = 500,
    seed: int = 0,
    horizon: int = 300,
    control_freq: int = 20,
    reward_shaping: bool = True,
) -> float:
    env = make_env(seed=seed, horizon=horizon, control_freq=control_freq, reward_shaping=reward_shaping, spurious_seed=seed)

    succ = 0.0
    for ep in range(int(episodes)):
        obs, info = env.reset(seed=seed + 10_000 + ep)
        done = False
        ep_succ = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_succ = max(ep_succ, float(info.get("success", 0.0)))
            done = bool(terminated) or bool(truncated)
        succ += ep_succ

    env.close()
    return float(succ / episodes)


def main():
    seed = 0

    # Paper Lift: horizon=300, control_freq=20, max training steps=1e6 :contentReference[oaicite:2]{index=2}
    total_steps = 1_000_000
    horizon = 300
    control_freq = 20
    reward_shaping = True

    eval_freq = 50_000
    eval_episodes = 50

    sac_kwargs: Dict[str, Any] = dict(
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,  # (paper used 128, but keep your baseline unchanged for now)
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
    )

    set_global_seed(seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"Lift_SAC_SB3_seed{seed}_h{horizon}_steps{total_steps}_NOMINAL_CONF_{timestamp}"
    run_dir = os.path.join("runs", run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tb_logs")
    ensure_dir(ckpt_dir)
    ensure_dir(tb_dir)

    train_env = make_env(seed=seed, horizon=horizon, control_freq=control_freq, reward_shaping=reward_shaping, spurious_seed=seed)
    eval_env  = make_env(seed=seed + 1, horizon=horizon, control_freq=control_freq, reward_shaping=reward_shaping, spurious_seed=seed + 1)

    obs_dim, act_dim = get_obs_act_dims(train_env)
    print("obs_dim:", obs_dim, "act_dim:", act_dim)  # expect obs_dim ~= 50

    config = {
        "run_id": run_id,
        "seed": seed,
        "total_steps": total_steps,
        "env": {
            "task": "Lift",
            "robot": "Panda",
            "obs": f"state({obs_dim})",
            "action_dim": act_dim,
            "horizon": horizon,
            "control_freq": control_freq,
            "reward_shaping": reward_shaping,
            "nominal_spurious": "left->green, right->red, RGB appended to obs",
        },
        "algo": {"name": "SAC", "library": "stable-baselines3", **sac_kwargs},
        "eval": {"eval_freq": eval_freq, "eval_episodes": eval_episodes, "report_eval_episodes": 500},
        "paths": {"run_dir": run_dir, "ckpt_dir": ckpt_dir, "tb_dir": tb_dir},
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    write_versions(os.path.join(run_dir, "versions.txt"))

    model = SAC("MlpPolicy", train_env, seed=seed, tensorboard_log=tb_dir, **sac_kwargs)

    best_cb = SuccessEvalCallback(eval_env=eval_env, eval_episodes=eval_episodes, eval_freq=eval_freq, save_dir=ckpt_dir, verbose=1)
    periodic_cb = CheckpointCallback(save_freq=eval_freq, save_path=ckpt_dir, name_prefix="ckpt", save_replay_buffer=False, save_vecnormalize=False)

    model.learn(total_timesteps=total_steps, callback=[best_cb, periodic_cb], tb_log_name=run_id)

    final_path = os.path.join(ckpt_dir, "final")
    model.save(final_path)
    print(f"✅ Saved final checkpoint: {final_path}.zip")

    best_path = os.path.join(ckpt_dir, "best_by_success.zip")
    if os.path.exists(best_path):
        best_model = SAC.load(best_path)
        sr500 = eval_success_rate(best_model, episodes=500, seed=seed + 999, horizon=horizon, control_freq=control_freq, reward_shaping=reward_shaping)
        print(f"✅ Best checkpoint success_rate over 500 eps: {sr500:.3f}")

        with open(os.path.join(run_dir, "eval_report.txt"), "w") as f:
            f.write(f"Best checkpoint: {best_path}\n")
            f.write(f"SR over 500 eps: {sr500:.6f}\n")
    else:
        print("⚠️ Could not find best_by_success.zip to evaluate.")

    train_env.close()
    eval_env.close()
    print(f"\nRun artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()

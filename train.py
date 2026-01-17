# train_sb3_sac_lift.py

import os
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from success import SuccessInfoWrapper  # <-- make sure this file exists


def make_env():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,      # state obs (simplest)
        control_freq=20,
        horizon=200,
        reward_shaping=True,
    )
    env = GymWrapper(env)            # Gymnasium API (reset->(obs,info), step->5-tuple)
    env = SuccessInfoWrapper(env)    # adds info["success"] (0/1)
    env = Monitor(env)               # logs ep reward/len for SB3
    return env


class SuccessEvalCallback(BaseCallback):
    """
    Evaluates success rate every eval_freq steps on eval_env,
    saves best model by success rate.
    """
    def __init__(self, eval_env, eval_episodes=20, eval_freq=20000, save_path="checkpoints", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_sr = -1.0

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
                done = terminated or truncated
            succ += ep_succ
        return succ / self.eval_episodes

    def _on_step(self) -> bool:
        # avoid eval at step 0
        if self.num_timesteps > 0 and (self.num_timesteps % self.eval_freq == 0):
            sr = self._evaluate()
            if self.verbose:
                print(f"\n[Eval] steps={self.num_timesteps} success_rate={sr:.3f}\n")

            if sr > self.best_sr:
                self.best_sr = sr
                best_path = os.path.join(self.save_path, "best_by_success_sb3")
                self.model.save(best_path)
                if self.verbose:
                    print(f"✅ Saved best checkpoint: {best_path}.zip (sr={sr:.3f})")

        return True


def eval_success_rate(model, episodes=50) -> float:
    env = make_env()
    succ = 0.0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_succ = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_succ = max(ep_succ, float(info.get("success", 0.0)))
            done = terminated or truncated
        succ += ep_succ
    env.close()
    return succ / episodes


def main():
    os.makedirs("checkpoints", exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
    )

    callback = SuccessEvalCallback(
        eval_env=eval_env,
        eval_episodes=20,
        eval_freq=20000,
        save_path="checkpoints",
        verbose=1,
    )

    total_steps = 200_000  # start with 200k; increase later
    model.learn(total_timesteps=total_steps, callback=callback)

    final_path = os.path.join("checkpoints", "final_sb3")
    model.save(final_path)
    print(f"✅ Saved final checkpoint: {final_path}.zip")

    sr = eval_success_rate(model, episodes=50)
    print(f"✅ Final eval success_rate over 50 eps: {sr:.3f}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

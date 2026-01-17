import os
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from success import SuccessInfoWrapper


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
    env = GymWrapper(env)
    env = SuccessInfoWrapper(env)  # adds info["success"]
    env = Monitor(env)             # logs episode reward/len
    return env


def eval_success_rate(model, episodes=50):
    env = make_env()
    succ = 0.0
    for ep in range(episodes):
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

    env = make_env()

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        # keep defaults mostly; these are safe for continuous control
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
    )

    # ---- SHORT TRAIN FIRST (prove it works) ----
    total_steps = 100_000
    model.learn(total_timesteps=total_steps)

    model.save("checkpoints/final_sb3.zip")
    print("✅ Saved checkpoints/final_sb3.zip")

    sr = eval_success_rate(model, episodes=50)
    print(f"✅ Eval success_rate over 50 eps: {sr:.3f}")

    env.close()


if __name__ == "__main__":
    main()

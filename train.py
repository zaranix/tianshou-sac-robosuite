import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper

def describe_obs(obs):
    if isinstance(obs, dict):
        print("obs is dict with keys:", list(obs.keys()))
        for k, v in obs.items():
            arr = np.asarray(v)
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    elif isinstance(obs, tuple):
        print("obs is tuple of length:", len(obs))
        for i, x in enumerate(obs):
            if isinstance(x, dict):
                print(f"  obs[{i}] is dict keys:", list(x.keys()))
            else:
                arr = np.asarray(x)
                print(f"  obs[{i}]: type={type(x)}, shape={arr.shape}, dtype={arr.dtype}")
    else:
        arr = np.asarray(obs)
        print("obs:", type(obs), "shape:", arr.shape, "dtype:", arr.dtype)

def make_env():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
        reward_shaping=True,
    )
    env = GymWrapper(env)
    return env

env = make_env()

reset_out = env.reset()
obs, info = reset_out  # gymnasium reset
print("obs shape:", obs.shape, "dtype:", obs.dtype)
print("reset info keys:", list(info.keys()))

for t in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(
        f"step={t}, reward={float(reward):.3f}, terminated={terminated}, truncated={truncated}, "
        f"info_keys={list(info.keys())[:8]}"
    )
    if done:
        obs, info = env.reset()
        print("Environment reset")
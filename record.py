import os
import numpy as np
import imageio.v2 as imageio

import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from success import SuccessInfoWrapper 

# ---------- Env factory ----------
def make_env(camera_name="frontview", width=640, height=480, gpu_id=0):
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,   # IMPORTANT
        use_camera_obs=False,
        render_camera=camera_name,
        render_gpu_device_id=gpu_id,
        control_freq=20,
        horizon=200,
        reward_shaping=True,
    )
    env = GymWrapper(env)
    env = SuccessInfoWrapper(env)
    env = Monitor(env)
    return env


# ---------- Helper: unwrap to robosuite base env with .sim ----------
def unwrap_to_sim(env):
    base = env
    while hasattr(base, "env") and not hasattr(base, "sim"):
        base = base.env
    if not hasattr(base, "sim"):
        raise RuntimeError("Could not find underlying robosuite env with .sim")
    return base.sim


# ---------- Helper: discover valid camera names ----------
def list_cameras(sim):
    # MuJoCo camera list
    cams = []
    for i in range(sim.model.ncam):
        name = sim.model.camera_id2name(i)
        if name is not None:
            cams.append(name)
    return cams


# ---------- Helper: render one frame ----------
def render_frame(sim, camera_name, width, height):
    frame = sim.render(width=width, height=height, camera_name=camera_name)
    frame = np.asarray(frame)
    # Some setups return upside-down; if yours is flipped, uncomment next line:
    # frame = np.flipud(frame)
    return frame


def record_episodes(
    ckpt_path,
    out_dir="videos",
    n_episodes=5,
    width=640,
    height=480,
    fps=20,
    gpu_id=0,
):
    os.makedirs(out_dir, exist_ok=True)

    # Start with a temp env to discover cameras
    tmp_env = make_env(camera_name="frontview", width=width, height=height, gpu_id=gpu_id)
    sim = unwrap_to_sim(tmp_env)
    cameras = list_cameras(sim)
    tmp_env.close()

    if not cameras:
        raise RuntimeError("No cameras found in sim.model (unexpected).")

    # Prefer common front cameras if available
    preferred = ["frontview", "agentview", "sideview", "birdview"]
    camera_name = None
    for c in preferred:
        if c in cameras:
            camera_name = c
            break
    if camera_name is None:
        camera_name = cameras[0]

    print("Available cameras:", cameras)
    print("Using camera:", camera_name)

    # Create final env with chosen camera
    env = make_env(camera_name=camera_name, width=width, height=height, gpu_id=gpu_id)
    sim = unwrap_to_sim(env)

    model = SAC.load(ckpt_path)

    results = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_succ = 0.0

        # Write to a temporary filename first
        tmp_video = os.path.join(out_dir, f"ep{ep:02d}_tmp.mp4")
        writer = imageio.get_writer(tmp_video, fps=fps)

        for t in range(200):  # horizon
            frame = render_frame(sim, camera_name=camera_name, width=width, height=height)
            writer.append_data(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_succ = max(ep_succ, float(info.get("success", 0.0)))
            done = terminated or truncated
            if done:
                break

        writer.close()

        # Rename video with success label
        final_video = os.path.join(out_dir, f"ep{ep:02d}_succ{int(ep_succ)}.mp4")
        os.replace(tmp_video, final_video)

        print(f"Episode {ep:02d}: success={ep_succ:.0f} saved={final_video}")
        results.append(ep_succ)

    env.close()

    sr = float(np.mean(results))
    print(f"\nSaved {n_episodes} videos to: {out_dir}")
    print(f"Success rate over recorded episodes: {sr:.3f}")


if __name__ == "__main__":
    # ✅ Update this path if needed
    CKPT = "checkpoints/best_by_success_sb3"
    record_episodes(CKPT, out_dir="videos", n_episodes=5, width=640, height=480, fps=20, gpu_id=0)

# spurious_lift.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NominalColorPosToStateWrapper(gym.Wrapper):
    """
    Paper Lift nominal spurious correlation:
      left (x_left)  -> GREEN
      right (x_right)-> RED

    IMPORTANT: because use_camera_obs=False (state-based),
    we append the cube RGB (3 floats) to the observation so the policy can "see" color.

    Resulting obs_dim = base_obs_dim + 3.
    """

    def __init__(
        self,
        env,
        *,
        x_left: float = -0.08,
        x_right: float = +0.08,
        y_center: float = 0.0,
        seed: int = 0,
    ):
        super().__init__(env)
        self.x_left = float(x_left)
        self.x_right = float(x_right)
        self.y_center = float(y_center)
        self.rng = np.random.RandomState(seed)

        # RGB for state (3 dims)
        self.GREEN_RGB = np.array([0.1, 0.9, 0.1], dtype=np.float32)
        self.RED_RGB   = np.array([0.9, 0.1, 0.1], dtype=np.float32)

        # RGBA for mujoco geom (4 dims)
        self.GREEN_RGBA = np.array([0.1, 0.9, 0.1, 1.0], dtype=np.float32)
        self.RED_RGBA   = np.array([0.9, 0.1, 0.1, 1.0], dtype=np.float32)

        self._last_rgb = self.GREEN_RGB.copy()

        # Expand observation_space by +3 for RGB
        if isinstance(env.observation_space, spaces.Box):
            low = np.concatenate([env.observation_space.low, np.array([0.0, 0.0, 0.0], dtype=np.float32)])
            high = np.concatenate([env.observation_space.high, np.array([1.0, 1.0, 1.0], dtype=np.float32)])
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # sample LEFT/RIGHT (0/1)
        u_pos = int(self.rng.randint(0, 2))
        x = self.x_left if u_pos == 0 else self.x_right

        # nominal mapping (paper):
        # left -> green, right -> red
        rgba = self.GREEN_RGBA if u_pos == 0 else self.RED_RGBA
        self._last_rgb = self.GREEN_RGB if u_pos == 0 else self.RED_RGB

        self._set_cube_pose_x(x)
        self._set_cube_color_rgba(rgba)

        # refresh observation AFTER changing sim (important!)
        obs = self._refresh_obs_if_possible(obs)

        return self._append_rgb(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._append_rgb(obs)
        return obs, reward, terminated, truncated, info

    # ---------- helpers ----------
    def _append_rgb(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self._last_rgb.astype(np.float32)], axis=0)

    def _refresh_obs_if_possible(self, obs):
        # If GymWrapper exposes an internal observation getter, use it.
        # (Different robosuite versions differ here.)
        if hasattr(self.env, "_get_observation"):
            try:
                return self.env._get_observation()
            except Exception:
                return obs
        if hasattr(self.env, "_get_obs"):
            try:
                return self.env._get_obs()
            except Exception:
                return obs
        return obs

    def _get_sim(self):
        # robust: gymnasium recommends env.unwrapped
        base = self.env.unwrapped
        if hasattr(base, "sim"):
            return base.sim
        # fallback unwrap chain
        base = self.env
        while hasattr(base, "env"):
            if hasattr(base, "sim"):
                return base.sim
            base = base.env
        return None

    def _set_cube_pose_x(self, x):
        sim = self._get_sim()
        if sim is None:
            return

        moved = False

        # Try moving via FREE joint qpos if available
        try:
            for jname in sim.model.joint_names:
                if "cube" in jname.lower():
                    jid = sim.model.joint_name2id(jname)
                    jnt_type = int(sim.model.jnt_type[jid])  # 0 = FREE
                    if jnt_type == 0:
                        adr = sim.model.jnt_qposadr[jid]
                        qpos = sim.data.qpos[adr:adr + 7].copy()
                        qpos[0] = x
                        qpos[1] = self.y_center
                        sim.data.qpos[adr:adr + 7] = qpos
                        sim.forward()
                        moved = True
                        break
        except Exception:
            moved = False

        # Fallback: body_pos for common robosuite names
        if not moved:
            for body_name in ("cube_main", "cube"):
                try:
                    bid = sim.model.body_name2id(body_name)
                    sim.model.body_pos[bid][0] = x
                    sim.model.body_pos[bid][1] = self.y_center
                    sim.forward()
                    break
                except Exception:
                    pass

    def _set_cube_color_rgba(self, rgba):
        sim = self._get_sim()
        if sim is None:
            return
        try:
            for gname in sim.model.geom_names:
                gl = gname.lower()
                if "cube" in gl:  # robust enough for Lift (you saw cube_g0)
                    gid = sim.model.geom_name2id(gname)
                    sim.model.geom_rgba[gid] = rgba
            sim.forward()
        except Exception:
            pass

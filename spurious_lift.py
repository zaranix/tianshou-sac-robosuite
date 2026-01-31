# spurious_lift.py
import numpy as np


class RobosuiteColorPosWrapper:
    """
    Wraps a *raw robosuite env* (NOT gymnasium) and:
      - sets cube x-position (left/right)
      - sets cube color (green/red)
    on every reset, then returns a refreshed observation dict.

    mode:
      - "confounded": left->green, right->red
      - "shifted_swapped": left->red, right->green  (swap mapping)
      - "shifted_indep": position and color independent
    """

    def __init__(
        self,
        env,
        mode="confounded",
        x_left=-0.08,
        x_right=+0.08,
        y_center=0.0,
        seed=0,
    ):
        self.env = env
        self.mode = mode
        self.x_left = float(x_left)
        self.x_right = float(x_right)
        self.y_center = float(y_center)
        self.rng = np.random.RandomState(seed)

        # store last chosen RGB (for the gym wrapper that appends RGB to obs)
        self.last_rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # RGB (not RGBA) for appending to observation
        self.GREEN_RGB = np.array([0.1, 0.9, 0.1], dtype=np.float32)
        self.RED_RGB   = np.array([0.9, 0.1, 0.1], dtype=np.float32)

        # RGBA for mujoco geom color
        self.GREEN_RGBA = np.array([0.1, 0.9, 0.1, 1.0], dtype=np.float32)
        self.RED_RGBA   = np.array([0.9, 0.1, 0.1, 1.0], dtype=np.float32)

    def __getattr__(self, name):
        # forward everything else to the wrapped robosuite env
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)  # robosuite returns obs dict
        self._apply_spurious_to_sim()
        # IMPORTANT: refresh obs AFTER moving/changing color
        if hasattr(self.env, "_get_observations"):
            obs = self.env._get_observations()
        return obs

    def step(self, action):
        return self.env.step(action)

    def _apply_spurious_to_sim(self):
        sim = getattr(self.env, "sim", None)
        if sim is None:
            return

        # sample pos confounder
        u_pos = int(self.rng.randint(0, 2))   # 0 left, 1 right
        u_col = int(self.rng.randint(0, 2))   # 0 green, 1 red (may be overwritten)

        if self.mode == "confounded":
            u_col = u_pos
        elif self.mode == "shifted_swapped":
            u_col = 1 - u_pos
        elif self.mode == "shifted_indep":
            pass
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        x = self.x_left if u_pos == 0 else self.x_right

        # --- Move cube: prefer body "cube_main" (your env showed this exists)
        moved = False
        try:
            bid = sim.model.body_name2id("cube_main")
            # If body has a FREE joint, we should move qpos. We do a robust search:
            jadr = sim.model.body_jntadr[bid]
            jnum = sim.model.body_jntnum[bid]
            for k in range(jnum):
                jid = jadr + k
                jtype = int(sim.model.jnt_type[jid])  # 0 = FREE
                if jtype == 0:
                    qpos_adr = sim.model.jnt_qposadr[jid]
                    qpos = sim.data.qpos[qpos_adr:qpos_adr + 7].copy()
                    qpos[0] = x
                    qpos[1] = self.y_center
                    sim.data.qpos[qpos_adr:qpos_adr + 7] = qpos
                    moved = True
                    break
            if moved:
                sim.forward()
        except Exception:
            moved = False

        # fallback: modify body_pos (less ideal but ok)
        if not moved:
            try:
                bid = sim.model.body_name2id("cube_main")
                sim.model.body_pos[bid][0] = x
                sim.model.body_pos[bid][1] = self.y_center
                sim.forward()
            except Exception:
                pass

        # --- Color the cube geom(s)
        rgba = self.GREEN_RGBA if u_col == 0 else self.RED_RGBA
        self.last_rgb = self.GREEN_RGB.copy() if u_col == 0 else self.RED_RGB.copy()
        try:
            for gname in sim.model.geom_names:
                gl = gname.lower()
                # your logs show geom name like "cube_g0"
                if "cube" in gl:
                    gid = sim.model.geom_name2id(gname)
                    sim.model.geom_rgba[gid] = rgba
            sim.forward()
        except Exception:
            pass


# -----------------------------
# Gymnasium wrapper to append RGB to flattened obs
# -----------------------------
import gymnasium as gym
import numpy as np


class AppendRGBObsWrapper(gym.Wrapper):
    """
    Appends 3 floats (RGB) to the end of the flattened state observation.

    It reads the RGB from the underlying robosuite wrapper (RobosuiteColorPosWrapper.last_rgb).
    Works only if that wrapper exists underneath.
    """
    def __init__(self, env):
        super().__init__(env)

        # extend observation space by 3 dims
        old = env.observation_space
        assert len(old.shape) == 1, f"Expected 1D obs, got {old.shape}"
        low = np.concatenate([old.low,  np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)])
        high = np.concatenate([old.high, np.array([ np.inf,  np.inf,  np.inf], dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_last_rgb(self):
        base = self.env
        while hasattr(base, "env"):
            if hasattr(base, "last_rgb"):
                return np.asarray(base.last_rgb, dtype=np.float32)
            base = base.env
        if hasattr(base, "last_rgb"):
            return np.asarray(base.last_rgb, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        rgb = self._get_last_rgb()
        obs = np.concatenate([np.asarray(obs, dtype=np.float32), rgb], axis=0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rgb = self._get_last_rgb()
        obs = np.concatenate([np.asarray(obs, dtype=np.float32), rgb], axis=0)
        return obs, reward, terminated, truncated, info

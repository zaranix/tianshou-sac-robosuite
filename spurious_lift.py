# spurious_lift.py
import numpy as np
import gymnasium as gym

class SpuriousColorPositionLiftWrapper(gym.Wrapper):
    """
    Injects spurious correlation between cube color and cube x-position in Robosuite Lift.

    mode:
      - "confounded": u controls BOTH x-pos and color (spurious correlation present)
      - "shifted_indep": x-pos and color are sampled independently (correlation broken)
      - "shifted_swapped": mapping between u->color is swapped vs training (breaks)
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
        super().__init__(env)
        self.mode = mode
        self.x_left = x_left
        self.x_right = x_right
        self.y_center = y_center
        self.rng = np.random.RandomState(seed)

        # RGBA
        self.green = np.array([0.1, 0.9, 0.1, 1.0])
        self.red   = np.array([0.9, 0.1, 0.1, 1.0])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_spurious()
        return obs, info

    def step(self, action):
        return self.env.step(action)

    # -----------------------------
    # Core logic (robosuite mujoco)
    # -----------------------------
    def _apply_spurious(self):
        # unwrap until robosuite env that has .sim
        base = self.env
        while hasattr(base, "env") and not hasattr(base, "sim"):
            base = base.env
        if not hasattr(base, "sim"):
            return

        sim = base.sim

        # Sample confounders
        u_pos = int(self.rng.randint(0, 2))
        u_col = int(self.rng.randint(0, 2))

        if self.mode == "confounded":
            u_col = u_pos
        elif self.mode == "shifted_swapped":
            u_col = 1 - u_pos
        elif self.mode == "shifted_indep":
            pass
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Position: left/right on x
        x = self.x_left if u_pos == 0 else self.x_right

        # --- Move cube: try free joint first ---
        moved = False
        try:
            for jname in sim.model.joint_names:
                if "cube" in jname.lower():
                    jid = sim.model.joint_name2id(jname)
                    qpos_adr = sim.model.jnt_qposadr[jid]
                    jnt_type = int(sim.model.jnt_type[jid])  # 0 = FREE
                    if jnt_type == 0:
                        qpos = sim.data.qpos[qpos_adr:qpos_adr + 7].copy()
                        qpos[0] = x
                        qpos[1] = self.y_center
                        sim.data.qpos[qpos_adr:qpos_adr + 7] = qpos
                        sim.forward()
                        moved = True
                        break
        except Exception:
            moved = False

        # --- Fallback: body_pos ---
        if not moved:
            try:
                bid = sim.model.body_name2id("cube")
                sim.model.body_pos[bid][0] = x
                sim.model.body_pos[bid][1] = self.y_center
                sim.forward()
            except Exception:
                pass

        # Color: set geom_rgba for geoms that look like cube/object
        rgba = self.green if u_col == 0 else self.red
        try:
            for gname in sim.model.geom_names:
                gl = gname.lower()
                if ("cube" in gl) or ("object" in gl) or ("box" in gl):
                    gid = sim.model.geom_name2id(gname)
                    sim.model.geom_rgba[gid] = rgba
            sim.forward()
        except Exception:
            pass

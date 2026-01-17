import gymnasium as gym

class SuccessInfoWrapper(gym.Wrapper):
    """
    Adds info['success'] = 0.0 or 1.0 using robosuite's internal _check_success().
    This is ONLY for evaluation/logging, not part of the observation.
    """
    def __init__(self, env):
        super().__init__(env)

    def _check_success_robosuite(self):
        base = self.env
        # unwrap until we find robosuite env
        while hasattr(base, "env"):
            if hasattr(base, "_check_success"):
                return float(base._check_success())
            base = base.env
        if hasattr(base, "_check_success"):
            return float(base._check_success())
        return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["success"] = self._check_success_robosuite()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["success"] = self._check_success_robosuite()
        return obs, reward, terminated, truncated, info


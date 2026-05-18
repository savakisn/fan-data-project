import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MaintenanceEnv(gym.Env):
    """
    One episode = one engine lifetime.
    State:   predicted RUL at the current cycle (single float).
    Actions: 0 = do nothing, 1 = replace.
    Rewards encode operational cost: late replacement is penalised harder
    than early replacement, and failure without replacement is catastrophic.
    """

    def __init__(self, episodes):
        # episodes: list of dicts with 'pred_rul' and 'actual_rul' (1-D float32 arrays)
        super().__init__()
        self.episodes = episodes
        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(500.0),
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self._ep   = None
        self._step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # random sampling so the agent sees varied degradation trajectories
        idx        = int(self.np_random.integers(0, len(self.episodes)))
        self._ep   = self.episodes[idx]
        self._step = 0
        return np.array([self._ep['pred_rul'][0]], dtype=np.float32), {}

    def step(self, action):
        pred_rul   = float(self._ep['pred_rul'][self._step])
        actual_rul = float(self._ep['actual_rul'][self._step])
        terminated = False
        truncated  = False

        if action == 1:
            # reward boundary at 30 cycles: close enough to failure to be intentional,
            # far enough to avoid an emergency
            reward     = 20.0 if actual_rul <= 30 else -10.0
            terminated = True
        else:
            if actual_rul <= 0:
                reward     = -100.0
                terminated = True
            else:
                reward      = 1.0
                self._step += 1
                if self._step >= len(self._ep['pred_rul']):
                    # safety net: real CMAPSS episodes always end on actual_rul == 0
                    terminated = True

        next_pred = 0.0 if terminated else float(self._ep['pred_rul'][self._step])
        return np.array([next_pred], dtype=np.float32), reward, terminated, truncated, {}

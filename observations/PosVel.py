# Direct measurement of position/velocity states

import numpy as np

from .Observation import Observation

class PosVel(Observation):
    def __init__(self, state_names, noise_cov, seed = 0):
        self.state_names = state_names
        self.obs_function = lambda x : x
        self.rng = np.random.default_rng(seed)
        self.noise_func = lambda x, cov : self.rng.multivariate_normal(x, cov)
        self.noise_params = noise_cov
        self.diff_type = 'custom'
        self.diff_func = lambda x : np.identity(x.size)
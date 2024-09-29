# Linear dynamics

import numpy as np

from .Dynamics import Dynamics

class DiscreteLinearDynamics(Dynamics):
    def __init__(self,Ffunc, Qfunc, Gfunc = None, seed = 0):
        if Gfunc is None:
            this_func = lambda state, input, dt : Ffunc(dt) @ state
        else:
            this_func = lambda state, input, dt : Ffunc(dt) @ state + Gfunc(dt) @ input
        self.dynamics_function = this_func
        self.solver_type = 'discrete'
        self.process_noise = lambda state, dt : Qfunc(dt)
        self.rng = np.random.default_rng(seed)
        self.noise_func = lambda x, cov : self.rng.multivariate_normal(x, cov)
        self.diff_type = 'autodiff'
        self.diff_func = lambda x, dt : Ffunc(dt)

    def propagateWithNoise(self, dt, state, input = None):
        new_state = self.propagate(dt, state, input)
        new_state = self.noise_func(new_state, self.process_noise(state, dt))
        return new_state

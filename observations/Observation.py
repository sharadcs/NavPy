# Template class for all observations

import autograd.numpy as np
from autograd import grad
from scipy import integrate as integrate

class Observation():
    def __init__(self, state_names, obs_function, noise_params, noise_func, diff_type, diff_func = None):
        self.state_names = state_names
        self.obs_function = obs_function
        self.noise_func = noise_func
        self.noise_params = noise_params
        self.diff_type = diff_type
        if self.diff_func is not None:
            self.diff_func = diff_func

    def getStateNames(self):
        return self.state_names

    def genMeas(self, state):
        return self.obs_function(state)
    
    def genMeasWithNoise(self, state):
        return self.noise_func(self.obs_function(state), self.noise_params)
    
    def jacobian(self, state):
        if self.diff_type == 'custom':
            J = self.diff_func(state)
        elif self.diff_type == 'autodiff' or self.diff_type == 'autograd':
            Jfunc = lambda x: self.obs_function(x)
            J = grad(Jfunc)(state)
        return J
    
    def noiseParams(self):
        return self.noise_params
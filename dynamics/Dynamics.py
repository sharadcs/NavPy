# Template class for all nonlinear dynamics

import autograd.numpy as np
from autograd import grad
from scipy import integrate as integrate

class Dynamics():
    def __init__(self, dynamics_function, process_noise, solver_type, diff_type, diff_func = None):
        self.dynamics_function = dynamics_function
        self.process_noise = process_noise
        self.solver_type = solver_type
        self.diff_type = diff_type
        if diff_func is not None:
            self.diff_func = diff_func

    def propagate(self, dt, state, input = None):
        new_state = np.full(np.shape(state),np.nan)
        if self.solver_type == 'discrete':
            new_state = self.dynamics_function(state, input, dt)
        elif self.solver_type == 'euler':
            new_state = state + dt * self.dynamics_function(state, input, dt)
        elif self.solver_type == 'rk45':
            new_state = np.array(integrate.solve_ivp(self.dynamics_function,dt,state.to_list(),method='RK45'))
        return new_state

    def jacobian(self, dt, state):
        if self.diff_type == 'custom':
            J = self.diff_func(state, dt)
        elif self.diff_type == 'autodiff' or self.diff_type == 'autograd':
            Jfunc = lambda x : self.dynamics_function(x, None, dt)
            J = grad(Jfunc)
            J = J(state)
        return J
    
    def noise(self, dt, state):
        return self.process_noise(state, dt)



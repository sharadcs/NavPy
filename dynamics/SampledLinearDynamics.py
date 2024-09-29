from Dynamics import Dynamics
import scipy.linalg as linalg
import numpy as np

class SampledLinearDynamics(Dynamics):
    def __init__(self, Amat, Qmat, Bmat = None, solver = 'exact'):
        if solver == 'euler' or solver == 'rk45':
            if Bmat is None:
                this_func = lambda state, input, dt : Amat@state
            else:
                this_func = lambda state, input, dt : Amat@state + Bmat@input
        elif solver == 'exact':
            if Bmat is None:
                this_func = lambda state, input, dt : linalg.expm(Amat*dt)@state
            else:
                this_func = lambda state, input, dt : linalg.expm(Amat*dt)@state + np.linalg.inv(Amat)@(np.identity(np.shape(Amat)[1])-linalg.expm(Amat*dt))@Bmat@input
        self.dynamics_function = this_func
        self.solver = solver
        self.process_noise = Qmat #TODO: add equations for integrating noise with/without ZOH assumption
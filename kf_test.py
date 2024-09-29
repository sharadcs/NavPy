import numpy as np

from dynamics import DiscreteLinearDynamics
from observations import PosVel
from navigation import Filter

state_list = [('posvel1',2),('posvel2',2)]
F = lambda dt : np.array([[1,dt],[0,1]])
Q = lambda dt : np.array([[0.01,0],[0,0.01]])
jac = lambda x, dt : F(dt)
#dyn_model = dyn.Dynamics(F, Q, 'discrete', 'custom', F)
dyn_model = DiscreteLinearDynamics(F, Q)
R = np.array([[0.1,0],[0,0.1]])
obs_model1 = PosVel(['posvel1'],R)
obs_model2 = PosVel(['posvel2'],R)

x_out1 = dyn_model.propagateWithNoise(1.,np.ones(2))
#print(x_out1)
x_out2 = dyn_model.propagateWithNoise(1.,np.ones(2))
#print(x_out2)
y_out1 = obs_model1.genMeasWithNoise(x_out1)
y_out2 = obs_model2.genMeasWithNoise(x_out2)
#print(y_out1)
y_out = np.append(y_out1,y_out2)

kf = Filter(state_list,np.ones(4),np.identity(4),[dyn_model,dyn_model],[obs_model1,obs_model2])
out_pred = kf.predict(1.)
#print(out_pred[0])
#print(out_pred[1])
out_corr = kf.update(y_out)
#print(out_corr[0])
#print(out_corr[1])
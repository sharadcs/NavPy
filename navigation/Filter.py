# Template Class for all filters (implements basic EKF)

import numpy as np
import scipy.linalg as linalg

class Filter():
    def __init__(self, state_list, state, cov, dynamics_list, observations_list, to_solve = None):
        self.state_list = state_list
        self.partitions = []
        for item in state_list:
            self.partitions += item[1]*[item[0]]
        self.state = np.reshape(state,(state.size,))
        self.cov = np.copy(cov)
        self.time = 0
        self.dynamics = dynamics_list
        self.observations = observations_list
        if to_solve is not None:
            self.consider = 1 - to_solve
        else:
            self.consider = np.zeros(state.size, dtype=bool)
        # TODO: self.history = 

    def setDynamics(self, dynamics_list):
        self.dynamics = dynamics_list

    def setObservations(self, observations_list):
        self.observations = observations_list

    def setStateCov(self, state, cov):
        self.state = state
        self.cov = cov

    def blockDiag(self, arr_list):
        size = sum(list(map(lambda x : np.shape(x)[0], arr_list)))
        mat = np.zeros((size,size))
        curr_ind = 0
        for arr in arr_list:
            block_len = arr.shape[0]
            mat[curr_ind:curr_ind+block_len,curr_ind:curr_ind+block_len] = arr
            curr_ind = curr_ind+block_len
        return mat

    def propagate(self, dt, input = None):
        pred_state = np.copy(self.state)
        Flist = []
        Qlist = []
        curr_ind = 0
        for state_info, obj in zip(self.state_list, self.dynamics):
            state_part = np.copy(self.state[curr_ind:curr_ind+state_info[1]])
            pred_state[curr_ind:curr_ind+state_info[1]] = obj.propagate(dt, state_part, input)
            curr_ind += state_info[1]
            Flist.append(obj.jacobian(dt, state_part))
            Qlist.append(obj.noise(dt, state_part))
        return (pred_state, Flist, Qlist)
            
    def genMeas(self):
        meas = np.array([])
        jacobian = np.array([])
        Rlist = []
        for obj in self.observations:
            state_names = obj.getStateNames()
            indices = list(filter(lambda x: self.partitions[x] in state_names, range(len(self.partitions))))
            state_part = self.state[indices]
            meas_part = obj.genMeas(state_part)
            meas = np.append(meas, meas_part)
            Rlist.append(obj.noiseParams())
            H_part = np.zeros((meas_part.size,self.state.size))
            H_part[:,indices] = obj.jacobian(state_part)
            if jacobian.size == 0:
                jacobian = H_part
            else:
                jacobian = np.vstack((jacobian,H_part))
        return (meas, jacobian, Rlist)

    def predictNoSave(self, dt, input = None):
        temp = self.propagate(dt, input)
        pred_state = temp[0]
        F = self.blockDiag(temp[1])
        pred_cov = F @ self.cov @ F.T + self.blockDiag(temp[2])
        return (pred_state, pred_cov)

    def predict(self, dt, input = None):
        temp = self.predictNoSave(dt, input)
        self.state = temp[0]
        self.cov = temp[1]
        self.time += dt
        return (self.state, self.cov)
    
    def update(self, meas):
        temp = self.genMeas()
        pred_meas = temp[0]
        H = temp[1]
        R = self.blockDiag(temp[2])
        innov = meas - pred_meas
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        K[self.consider] = 0
        updated_state = self.state + K @ innov
        eye = np.identity(self.state.size)
        updated_cov = (eye - K @ H) @ self.cov @ (eye - K @ H).T + K @ R @ K.T
        self.state = updated_state
        self.cov = updated_cov
        return (self.state, self.cov)

    def evaluateMeas(self, dt, meas, input = None):
        temp = self.predictNoSave(dt, input)
        pred_state = temp[0]
        pred_cov = temp[1]
        innov = meas - self.observations.genMeas(self.state)
        H = self.observations.jacobian(self.state)
        R = self.observations.noiseParams(self.state)
        S = H @ self.cov @ H.T + R
        return innov.T @ np.linalg.inv(S) @ innov

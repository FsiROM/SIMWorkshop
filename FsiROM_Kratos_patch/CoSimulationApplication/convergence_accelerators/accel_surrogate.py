import numpy as np
import torchvision
import torch


class accel_SURROG:

    def __init__(self, ):
        pass

    def train(self, label_data=None, r_orth=None, subsp_dims=None, subsp_dists=None, file_name=None):

        self.label_norm = np.linalg.norm(label_data, axis = 1)[:, np.newaxis]
        self.r_orth_norm = np.linalg.norm(r_orth, axis = 1)[:, np.newaxis]
        self.subsp_dists_norm = np.linalg.norm(subsp_dists)

        if file_name is not None:
            #self.nn_model = AutEnc()
            self.nn_model.load_state_dict(torch.load(file_name))
            self.nn_model.eval();
            self.nn_model = self.nn_model.double()

    def pred(self, delta_r_orth, dim, dist):

        nrm_delta_r_orth = torch.tensor(delta_r_orth/self.r_orth_norm).T       
        dist_nrm = (dist)/(self.subsp_dists_norm)
        subsp_data2 = torch.tensor(dist_nrm)
        subsp_data = subsp_data2.reshape((-1, 1))
        input_data = torch.hstack((subsp_data, nrm_delta_r_orth))
        first_infer = self.nn_model.infer(input_data)

        inferred =  first_infer * nrm_delta_r_orth

        return inferred.detach().numpy().T * self.label_norm

    def pred_deltax(self, delta_r_orth, dim, dist):
        
        res = self.pred(delta_r_orth, dim, dist)

        return res

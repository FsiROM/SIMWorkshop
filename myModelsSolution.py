import numpy as np
from rom_am.pod import POD
from rom_am.rom import ROM
from rom_am.dimreducers.rom_DimensionalityReducer import *


class MyForceReducer(RomDimensionalityReducer):

    def __init__(self, latent_dim, ) -> None:
        super().__init__(latent_dim)

    def train(self, data, map_used=None, normalize=True, center=True):

        super().train(data, map_used)

        pod = self._call_POD_core()
        rom = ROM(pod)
        rom.decompose(X=data, normalize=normalize, center=center,
                      rank=self.latent_dim)

        self.latent_dim = pod.kept_rank
        self.normalize = normalize
        self.center = center
        self.rom = rom
        self.pod = pod

    def encode(self, new_data):

        interm = self.rom.normalize(self.rom.center(new_data))
        return self.pod.project(interm)

    def _call_POD_core(self, ):
        return POD()

    @property
    def reduced_data(self):
        return self.pod.pod_coeff

import numpy as np


class MyDispReducer(MyForceReducer):

      def decode(self, new_data, high_dim=False):

        interm = self.pod.inverse_project(new_data)
        return self.rom.decenter(self.rom.denormalize(interm))

from sklearn.linear_model import LassoLarsIC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from rom_am.regressors.rom_regressor import *


class MyRegressor(RomRegressor):

    def __init__(self, poly_degree=2, criterion='bic', intercept_=True) -> None:
        super().__init__()
        self.criterion = criterion
        self.poly_degree = poly_degree
        self.intercept_ = intercept_

    def train(self, input_data, output_data):
        super().train(input_data, output_data)

        self.regr_model = make_pipeline(
            PolynomialFeatures(self.poly_degree, include_bias=self.intercept_), MultiOutputRegressor(LassoLarsIC(criterion=self.criterion),))
        self.regr_model.fit(input_data.T, output_data.T)

        self.nonzeroIds = []
        for i in range(self.output_dim):
            self.nonzeroIds.append(np.argwhere(np.abs(
                self.regr_model["multioutputregressor"].estimators_[i].coef_) > 1e-9)[:, 0])

    def predict(self, new_input):

        # Instead of self.regr_model.predict(new_input.T).T, the following is faster :
        self.polyFeatures = self.regr_model["polynomialfeatures"].transform(
            new_input.T)

        def mult_(proc):
            linear_ = self.polyFeatures[:, self.nonzeroIds[proc]] @ self.regr_model["multioutputregressor"].estimators_[
                proc].coef_[self.nonzeroIds[proc]].reshape((-1, 1))
            return linear_ + self.regr_model["multioutputregressor"].estimators_[proc].intercept_

        res = np.empty((self.output_dim, new_input.shape[1]))
        for i in range(self.output_dim):
            res[i, :] = mult_(i).ravel()

        return res

class MyMappedDispReducer(MyDispReducer):

  def train(self, data, map_used=None, normalize=True, center=True):
    super().train(data, map_used=map_used, normalize=normalize, center=center)

    if map_used is not None:
        self.interface_dim = map_used.shape[0]
        self.map_mat = map_used
        self.inverse_project_mat = self.map_mat @ self.rom.denormalize(
            self.pod.modes)

        if center:
            self.mapped_mean_flow = self.map_mat @ self.rom.mean_flow.reshape(
                (-1, 1))

  def decode(self, new_data, high_dim=False):

      if self.map_mat is not None and not high_dim:
          interm = self._mapped_decode(new_data)
          if self.center:
              interm = (
                  interm + self.mapped_mean_flow).reshape((-1, new_data.shape[1]))
          return interm

      else:

          interm = self.pod.inverse_project(new_data)
          return self.rom.decenter(self.rom.denormalize(interm))
          
  def _mapped_decode(self, new_data):
        return self.inverse_project_mat @ new_data



import KratosMultiphysics as KM
from KratosMultiphysics.CoSimulationApplication.co_Romsimulation_analysis import CoRomSimulationAnalysis

"""
For user-scripting it is intended that a new class is derived
from CoSimulationAnalysis to do modifications
Check also "kratos/python_scripts/analysis_stage.py" for available methods that can be overridden
"""

import numpy as np
from rom_am.quad_man import QUAD_MAN
from rom_am.dimreducers.rom_am.podReducer import PodReducer


class MYFUNCT(PodReducer):

    def train(self, data, map_used=None, normalize=True, center=True):
        super().train(data, map_used, normalize, center)
        if map_used is not None:
            self.inverse_project_Vbar = self.map_mat @ self.rom.denormalize(
                self.pod.Vbar)

    def _call_POD_core(self):
        return QUAD_MAN()

    def _mapped_decode(self, new_data):
        return super()._mapped_decode(new_data) + self.inverse_project_Vbar @ self.pod._kron_x_sq(new_data)


parameter_file_name = "ProjectParametersCoSim.json"
with open(parameter_file_name,'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

simulation = CoRomSimulationAnalysis(parameters, outputReduc_model=MYFUNCT(.9999), inputReduc_model=PodReducer(24))
simulation.Run()

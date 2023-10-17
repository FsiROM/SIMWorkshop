import KratosMultiphysics as KM
from KratosMultiphysics.CoSimulationApplication.co_Romsimulation_analysis import CoRomSimulationAnalysis
from myModels import *

timepredictionOnly = False

parameter_file_name = "ProjectParametersCoSim.json"
with open(parameter_file_name, 'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

if not timepredictionOnly:
    simulation = CoRomSimulationAnalysis(parameters,)
else:
    disp_latentDim = .9999
    force_latentDim = 24
    simulation = CoRomSimulationAnalysis(parameters, inputReduc_model=MyForceReducer(force_latentDim),
                                         regression_model=MyRegressor(),
                                         outputReduc_model=MyDispReducer(disp_latentDim))
simulation.Run()

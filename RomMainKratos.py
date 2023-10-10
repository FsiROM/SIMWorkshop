import KratosMultiphysics as KM
from KratosMultiphysics.CoSimulationApplication.co_Romsimulation_analysis import CoRomSimulationAnalysis
from myModels import *


"""
For user-scripting it is intended that a new class is derived
from CoSimulationAnalysis to do modifications
Check also "kratos/python_scripts/analysis_stage.py" for available methods that can be overridden
"""

parameter_file_name = "ProjectParametersCoSim.json"
with open(parameter_file_name,'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

simulation = CoRomSimulationAnalysis(parameters,)
simulation.Run()

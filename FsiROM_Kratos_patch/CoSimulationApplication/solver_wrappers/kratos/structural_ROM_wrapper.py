# Importing the Kratos Library
import KratosMultiphysics as KM
import KratosMultiphysics.CoSimulationApplication.colors as colors

# Importing the base class
from KratosMultiphysics.CoSimulationApplication.solver_wrappers.kratos import structural_mechanics_wrapper
from KratosMultiphysics import StructuralMechanicsApplication

# Other imports
import numpy as np
from collections import deque
from rom_am.solid_rom import solid_ROM
import KratosMultiphysics.CoSimulationApplication.co_simulation_tools as cs_tools
import time


def Create(settings, model, solver_name):
    return StructuralROMWrapper(settings, model, solver_name)


class StructuralROMWrapper(structural_mechanics_wrapper.StructuralMechanicsWrapper):

    def __init__(self, settings, model, solver_name):

        super().__init__(settings, model, solver_name)

        self.ModelPart = self._analysis_stage._GetSolver().GetComputingModelPart()
        self.x0_vec = KM.VariableUtils().GetInitialPositionsVector(self.ModelPart.Nodes,2)
        self.initialize_data()
        self.get_rom_settings()
        self.trained = False
        self.map_used = None
        self._already_recievedData = False
        self.rom_model = None

    def receive_input_data(self, input_data):

        self.current_load = input_data.reshape((-1, 1))
        self._already_recievedData = True

        if self.is_in_collect_data_mode():
            self.update_load_data(self.current_load)
        if self.save_tr_data:
            np.save("./coSimData/load_data.npy",
                    np.asarray(self.load_data)[:, :, 0].T)

    def get_rom_settings(self, ):
        self.launch_time = self.settings["launch_time"].GetDouble()
        self.start_collecting_time = self.settings["start_collecting_time"].GetDouble()
        self.imported_model = self.settings["imported_model"].GetBool()
        self.save_model = self.settings["save_model"].GetBool()
        self.input_data_name = self.settings["input_data"]["data"].GetString()
        self.output_data_name = self.settings["output_data"]["data"].GetString()
        self.interface_only = self.settings["interface_only"].GetBool()
        self.use_map = self.settings["use_map"].GetBool()
        self.save_tr_data = self.settings["save_training_data"].GetBool()
        self.force_norm_regr = self.settings["force_norm_regr"].GetBool()
        self.disp_norm_regr = self.settings["disp_norm_regr"].GetBool()
        self.force_norm = self.settings["force_norm"].GetString()
        self.disp_norm = self.settings["disp_norm"].GetString()

    def initialize_data(self, ):
        self.load_data = deque()
        self.displacement_data = deque()
        self.displacement_data2 = deque()
        self.recons_time = []

    def is_in_prediction_mode(self, ):
        return self.prediction_mode_strategy()

    def is_in_collect_data_mode(self, ):
        return self.data_collect_mode_strategy()

    def prediction_mode_strategy(self, ):
        # ======= Condition to be met for launching ROM prediction ============
        return self._analysis_stage.time >= self.launch_time

    def data_collect_mode_strategy(self, ):
        # ======= Condition to be met for collecting training data ============
        return self._analysis_stage.time < self.launch_time
    
    def train_rom(self):

        if not self.trained:
            self.rom_model = solid_ROM()

            # ======= Import a trained ROM model ============
            if self.imported_model:
                file = self.settings["file"]
                import pickle
                with open(file["file_name"].GetString(), 'rb') as inp:
                    self.rom_model = pickle.load(inp)

            # ======= Train a ROM model ============
            else:
                #coords = np.asarray(self.GetInterfaceData(self.output_data_name).model_part.GetNodes())[:, :2]
                self.rom_model.train(np.asarray(self.load_data)[:, :, 0].T, np.asarray(self.displacement_data)[:, :, 0].T, 
                                     rank_pres=24, rank_disp=.9999,
                                     map_used = self.map_used,
                                     norm_regr=[self.force_norm_regr, self.disp_norm_regr],
                                     norm=[self.force_norm, self.disp_norm], 
                                     forcesReduc_model=self.inputReduc_model, regression_model=self.regression_model,
                                     dispReduc_model=self.outputReduc_model)

                # ======= Save ROM model in a file ============
                if self.save_model:
                    file = self.settings["file"]
                    import pickle
                    with open(file["file_name"].GetString(), 'wb') as outp:
                        pickle.dump(self.rom_model, outp,
                                    pickle.HIGHEST_PROTOCOL)
            self.trained = True
            self.last_fom_step = self.ModelPart.ProcessInfo[KM.STEP]
        else:
            pass

    def rom_output(self, current_load):
        return self.rom_model.pred(current_load)

    def update_load_data(self, current_load):
        self.load_data.append(current_load)
        if self.save_tr_data:
            np.save("./coSimData/load_data.npy",
                    np.asarray(self.load_data)[:, :, 0].T)
            
    def update_disp_data(self, current_disp):
        self.displacement_data.append(current_disp)
        self.displacement_data2.append(self.GetInterfaceData(
                        self.output_data_name).GetData().reshape((-1, 1)))
        if self.save_tr_data:
            np.save("./coSimData/disp_data.npy",
                    np.asarray(self.displacement_data)[:, :, 0].T)
            np.save("./coSimData/disp_interf_data.npy",
                    np.asarray(self.displacement_data2)[:, :, 0].T)
            
    def SolveSolutionStep(self):
        # ======= Store Load Training Data ============
        if not self.is_in_prediction_mode() and self.is_in_collect_data_mode() and not self._already_recievedData:
            if not self.imported_model:
                current_load = self.GetInterfaceData(
                    self.input_data_name).GetData().reshape((-1, 1))
                self.update_load_data(current_load)

        # ======= Predict using the FOM ============
        # print("\n before slv \n")
        # conditions_array=self.mP.GetSubModelPart("PointLoad2D_FsiPoints11").Conditions
        # conditions_array=self.ModelPart.Conditions
        # for i, condition in enumerate(conditions_array):
        #     thisId = condition.GetNode(0).Id
        #     print("\n cnd PT LOAD ", thisId, " ", np.asarray(condition.GetValue(StructuralMechanicsApplication.POINT_LOAD)), "\n")
        if not self.is_in_prediction_mode():
            super().SolveSolutionStep()

        # ======= Predict using the ROM ============
        else:
            self.train_rom()
            if self._already_recievedData:
                current_load = self.current_load
            else:
                current_load = self.GetInterfaceData(
                    self.input_data_name).GetData().reshape((-1, 1))
            predicted_disp = self.rom_output(current_load).ravel()

            # ======= Predict The interface displacement only ============
            if self.interface_only:
                self.GetInterfaceData(self.output_data_name).SetData(predicted_disp)

            # ======= Predict The full displacement field ============
            else:
                #super().SolveSolutionStep()
                # tmp = KM.VariableUtils().GetSolutionStepValuesVector(
                #         self._analysis_stage._GetSolver().GetComputingModelPart().Nodes, KM.DISPLACEMENT, 0, 2)

                if self.use_map:
                    disp_arr = np.empty((self.SS, ))
                    disp_arr[self.ids_] = predicted_disp
                    KM.VariableUtils().SetSolutionStepValuesVector(self.ModelPart.Nodes,
                                                                KM.DISPLACEMENT, 1.*disp_arr, 0)
                else:
                    KM.VariableUtils().SetSolutionStepValuesVector(self.ModelPart.Nodes,
                                                                KM.DISPLACEMENT, 1.*predicted_disp, 0)
                    x_vec = self.x0_vec + 1.*predicted_disp
                    KM.VariableUtils().SetCurrentPositionsVector(self.ModelPart.Nodes,1.*x_vec)
                self.ModelPart.GetCommunicator().SynchronizeVariable(KM.DISPLACEMENT)

                # c = 0
                # for node in self.ModelPart.Nodes:
                #     id_ = node.Id
                #     which_id = np.argwhere(self.ids_global == id_)[0][0]
                #     node.SetValue(KM.DISPLACEMENT, [predicted_disp[2*which_id], predicted_disp[2*which_id+1], 0.])
                #     c+=1

                # for i, condition in enumerate(conditions_array):
                #     thisId = condition.GetNode(0).Id
                #     intrfIndex = np.argwhere(ids_local == thisId)
                #     if len(intrfIndex) == 0:
                #         pass
                #     else:
                #         intrfIndex = intrfIndex[0][0]
                #         this_force = [-current_load[2*intrfIndex], -current_load[2*intrfIndex+1]]
                #         tmp = condition.GetValue(StructuralMechanicsApplication.POINT_LOAD)
                #         tmp[0] = this_force[0]
                #         tmp[1] = this_force[1]
                #         condition.SetValue(StructuralMechanicsApplication.POINT_LOAD, tmp)

        # ======= Store Displacement Training Data ============
        self._already_recievedData = False # Resetting receiving data
        if not self.is_in_prediction_mode() and self.is_in_collect_data_mode():
            if not self.imported_model:
                if self.interface_only:
                    current_disp = self.GetInterfaceData(
                        self.output_data_name).GetData().reshape((-1, 1))
                else:
                    current_disp = KM.VariableUtils().GetSolutionStepValuesVector(
                        self.ModelPart.Nodes, KM.DISPLACEMENT, 0, 2)
                    current_disp = np.array(current_disp).reshape((-1, 1))
                self.update_disp_data(current_disp)

        # # ========= Compute the residuals ===================
        # residuals = KM.Vector(self.SS)
        # for i in range(self.SS):
        #     residuals[i] = 0.0
        # Assemble the global residual vector
        # self.BS.BuildRHS(self.SCH, self.ModelPart, residuals)
        # for i in range(self.SS):
        #     residuals[i] += KM.VariableUtils().GetSolutionStepValuesVector(
        #                 self.ModelPart.Nodes, StructuralMechanicsApplication.POINT_LOAD, 0, 2)[i]
        # print("\n", self.ModelPart, "\n")
        # print("\n", np.linalg.norm(np.asarray(residuals)), "\n")

        # conditions_array=model_part.GetSubModelPart("Parts_Solid_Structure").Conditions
        # current_load = self.GetInterfaceData(
        #             self.input_data_name).GetData()
        # self.ids_local = []
        # for node in self.GetInterfaceData(self.input_data_name).model_part.GetNodes():
        #     self.ids_local.append(node.Id)
        # ids_local = np.array(self.ids_local)
        # print("\n Prescribed LOAD norm", np.linalg.norm(KM.VariableUtils().GetSolutionStepValuesVector(
        #                self._analysis_stage._GetSolver().GetComputingModelPart().Nodes, StructuralMechanicsApplication.POINT_LOAD, 0, 2)), " \n")
        # conditions_array=model_part.GetSubModelPart("Parts_Solid_Structure").Conditions
        # for i, condition in enumerate(conditions_array):
        #     thisId = condition.GetNode(0).Id
        #     intrfIndex = np.argwhere(ids_local == thisId)
        #     print("\n cnd PT LOAD ", thisId, " ", condition.GetValue(StructuralMechanicsApplication.POINT_LOAD), "\n")

    def Initialize(self):
        super().Initialize()
        np.save("./coSimData/coords_interf.npy", 
                np.asarray(self.GetInterfaceData(self.output_data_name).model_part.GetNodes())[:, :2])

        # model_part = self._analysis_stage._GetSolver().GetComputingModelPart().GetSubModelPart("StructureInterface2D_Struc_Fsi")
        # model_part.GetSubModelPart("PointLoad2D_FsiPoints11")
        self.SS = self.ModelPart.GetCommunicator().GetDataCommunicator().Sum(self.ModelPart.NumberOfNodes() * 2, 0)
        self.SCH = self._analysis_stage._GetSolver()._GetScheme()
        # # Get the BuilderAndSolver instance
        self.BS = self._analysis_stage._GetSolver()._GetBuilderAndSolver()
        #self.BS.Check(self.ModelPart)
        #self.BS.SetUpDofSet(self.SCH, self.ModelPart)
        #self.BS.SetUpSystem(self.ModelPart)

        if self.save_tr_data or self.use_map:
            self.saved_out_t = []
            self.ids_local = []
            self.ids_global = []
            for node in self.GetInterfaceData(self.output_data_name).model_part.GetNodes():
                self.ids_local.append(node.Id)
            for node in self.ModelPart.Nodes:
                self.ids_global.append(node.Id)
            self.ids_ = np.in1d(self.ids_global, 
                                self.ids_local).nonzero()[0]
            c = np.empty((2*self.ids_.size,), dtype=self.ids_.dtype)
            c[::2] = 2*self.ids_
            c[1::2] = 2*self.ids_+1
            self.ids_ = c
            map_used = np.zeros((2*len(self.ids_global), len(self.ids_)))
            map_used[self.ids_, :] = np.eye(len(self.ids_), len(self.ids_))
            map_used = map_used.T
            np.save("./coSimData/map_used.npy", map_used)
            if self.use_map:
                self.map_used = map_used

            self.ids_global = np.array(self.ids_global)

    def FinalizeSolutionStep(self,):
        if self.use_map and self.is_in_prediction_mode():
            self.rom_model.store_last_result()
            self.saved_out_t.append(self._analysis_stage.time)
        else:
            super().FinalizeSolutionStep()

    def _GetNewSimulationName(self, ):
        return "::["+colors.yellow("Structural ROM")+"]:: "

    def InitializeSolutionStep(self,):
        if self.is_in_prediction_mode():
            KM.Logger.PrintInfo(self._GetNewSimulationName(), "STEP: ", self.ModelPart.ProcessInfo[KM.STEP])
            KM.Logger.PrintInfo(self._GetNewSimulationName(), "TIME: ", self._analysis_stage.time)
        if self.use_map and self.is_in_prediction_mode():
            pass
        else:
            super().InitializeSolutionStep()

    def Finalize(self):
        if self.use_map and self.rom_model is not None:
            self.u = self.rom_model.return_big_disps()
            self.export_results()

        super().Finalize()

    def OutputSolutionStep(self):
        if not (self.is_in_prediction_mode() and (self.use_map or self.interface_only)):
            super().OutputSolutionStep()


    def export_results(self):
        cs_tools.cs_print_info("Exporting fields on the complete physical domain")

        x0_vec = KM.VariableUtils().GetInitialPositionsVector(self.ModelPart.Nodes,2)
        t0 = time.time()
        for i in range(self.u.shape[1]):

            self.ModelPart.ProcessInfo[KM.STEP] = i + self.last_fom_step
            KM.VariableUtils().SetSolutionStepValuesVector(self.ModelPart.Nodes,
                                                KM.DISPLACEMENT, 1.*self.u[:, i], 0)
            self.ModelPart.GetCommunicator().SynchronizeVariable(KM.DISPLACEMENT)

            x_vec = x0_vec + 1.*self.u[:, i]
            KM.VariableUtils().SetCurrentPositionsVector(self.ModelPart.Nodes,1.*x_vec)
            super().OutputSolutionStep()
        t1 = time.time()
        self.recons_time.append(t1 - t0)
        with open("./coSimData/structure_recons_time.npy", 'wb') as f:
            np.save(f, np.array(self.recons_time))

    def ReceiveRomComponents(self, inputReduc_model=None, regression_model=None, outputReduc_model=None):
        self.inputReduc_model = inputReduc_model
        self.regression_model = regression_model
        self.outputReduc_model = outputReduc_model

    @classmethod
    def _GetDefaultParameters(cls):
        return KM.Parameters("""{
            "type"                    : "",
            "solver_wrapper_settings" : {},
            "io_settings"             : {},
            "data"                    : {},
            "mpi_settings"            : {},
            "echo_level"              : 0,
            "launch_time"             : 100.0,
            "force_norm_regr"         : false,
            "disp_norm_regr"          : false,
            "force_norm"              : "l2",
            "disp_norm"               : "l2",
            "start_collecting_time"   : 0.0,
            "imported_model"          : false,
            "save_model"              : false,
            "input_data"              : {},
            "output_data"             : {},
            "interface_only"          : false,
            "use_map"                 : true,
            "file"                    : {},
            "save_training_data"      : false
        }""")

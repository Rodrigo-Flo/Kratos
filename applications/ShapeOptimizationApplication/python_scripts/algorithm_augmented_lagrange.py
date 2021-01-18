# ==============================================================================
#  KratosShapeOptimizationApplication
#
#  License:         BSD License
#                   license: ShapeOptimizationApplication/license.txt
#
#  Main authors:    Baumgaertner Daniel, https://github.com/dbaumgaertner
#                    Flores Flores Rodrigo
#
# ==============================================================================

# Making KratosMultiphysics backward compatible with python 2.6 and 2.7
from __future__ import print_function, absolute_import, division

# Kratos Core and Apps
import KratosMultiphysics as KM
import KratosMultiphysics.ShapeOptimizationApplication as KSO

# Additional imports
from .algorithm_base import OptimizationAlgorithm
from . import mapper_factory
from . import data_logger_factory
import math
from . import custom_math as cm
from .custom_timer import Timer
from .custom_variable_utilities import WriteDictionaryDataOnNodalVariable, ReadNodalVariableToList, WriteListToNodalVariable
import copy

# ==============================================================================
class AlgorithmAugmentedLagrange(OptimizationAlgorithm):
    # --------------------------------------------------------------------------
    def __init__(self, optimization_settings, analyzer, communicator, model_part_controller):
        default_algorithm_settings = KM.Parameters("""
        {
            "name"                    : "augmented_lagrange",
            "max_correction_share"    : 0.75,
            "max_total_iterations"    : 10000,
            "max_outer_iterations"    : 10000,
            "max_inner_iterations"    : 30,
            "min_inner_iterations"    :  3,
            "inner_iteration_tolerance"      : 1e-3,
            "line_search" : {
                "line_search_type"           : "manual_stepping",
                "normalize_search_direction" : true,
                "step_size"                  : 0.5
            }
        }""")
        #Optimization 
        self.algorithm_settings =  optimization_settings["optimization_algorithm"]
        self.algorithm_settings.RecursivelyValidateAndAssignDefaults(default_algorithm_settings)


        self.optimization_settings = optimization_settings
        self.mapper_settings = optimization_settings["design_variables"]["filter"] #Filter applied on vertex morphing
        #Filter for penalty term?
        '''if self.algorithm_settings["filter_penalty_term"].GetBool():
            if self.algorithm_settings["penalty_filter_radius"].GetDouble() == -1.0:
                raise RuntimeError("The parameter `penalty_filter_radius` is missing in order to filter the penalty term!")'''
        #analyzer, communicator, model_part_controller necessary for the optimizer
        self.analyzer = analyzer
        self.communicator = communicator
        self.model_part_controller = model_part_controller

        self.design_surface = None
        self.mapper = None
        self.data_logger = None
        self.optimization_utilities = None

        self.objectives = optimization_settings["objectives"]
        self.constraints = optimization_settings["constraints"]
        
        self.lambda_g_0=[]
        self.lambda_h_0=[]
        self.p_vect_ineq_0=[]
        self.p_vect_eq_0=[]
        self.number_ineq=0
        self.number_eq=0
        self.p=1.0
        self.pmax=1e+10*self.p
        

        self.constraint_gradient_variables = {}
        for itr, constraint in enumerate(self.constraints):
            self.constraint_gradient_variables.update({
                constraint["identifier"].GetString() : {
                    "gradient": KM.KratosGlobals.GetVariable("DC"+str(itr+1)+"DX"),
                    "mapped_gradient": KM.KratosGlobals.GetVariable("DC"+str(itr+1)+"DX_MAPPED")
                }
            })

        #Bead optimization
        """
        self.bead_height = self.algorithm_settings["bead_height"].GetDouble()
        self.bead_side = self.algorithm_settings["bead_side"].GetString()
        self.filter_penalty_term = self.algorithm_settings["filter_penalty_term"].GetBool()
        self.estimated_lagrange_multiplier = self.algorithm_settings["estimated_lagrange_multiplier"].GetDouble()
        
        
       
        self.max_correction_share = self.algorithm_settings["max_correction_share"].GetDouble()
        """

        self.step_size = self.algorithm_settings["line_search"]["step_size"].GetDouble()
        #self.max_iterations = self.algorithm_settings["max_iterations"].GetInt() + 1
        self.max_total_iterations = self.algorithm_settings["max_total_iterations"].GetInt()
        self.max_outer_iterations = self.algorithm_settings["max_outer_iterations"].GetInt()
        
        self.max_inner_iterations = self.algorithm_settings["max_inner_iterations"].GetInt()
        self.min_inner_iterations = self.algorithm_settings["min_inner_iterations"].GetInt()
        self.inner_iteration_tolerance = self.algorithm_settings["inner_iteration_tolerance"].GetDouble()
        
        #self.relative_tolerance = self.algorithm_settings["relative_tolerance"].GetDouble()

        self.optimization_model_part = model_part_controller.GetOptimizationModelPart()
        
        #Variables for steepest descendent
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.SEARCH_DIRECTION)
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.CORRECTION)

        #Bead optimization variables
        """
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.ALPHA)
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.ALPHA_MAPPED)
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.DF1DALPHA)
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.DF1DALPHA_MAPPED)
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.DPDALPHA)
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.DPDALPHA_MAPPED)
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.DLDALPHA)
        """
        
    #  How many objective and constraints supports your method?--------------------------------------------------------------------------
    def CheckApplicability(self):
        if self.objectives.size() > 1:
            raise RuntimeError("Augmented Lagrange Method algorithm only supports one objective function at the moment!")

    # --------------------------------------------------------------------------
    def InitializeOptimizationLoop(self):
        self.model_part_controller.Initialize()
        self.model_part_controller.SetMinimalBufferSize(2)
        
        self.analyzer.InitializeBeforeOptimizationLoop()
        self.design_surface = self.model_part_controller.GetDesignSurface()
        

        self.mapper = mapper_factory.CreateMapper(self.design_surface, self.design_surface, self.mapper_settings)
        self.mapper.Initialize()
        """
        if self.filter_penalty_term:
            penalty_filter_radius = self.algorithm_settings["penalty_filter_radius"].GetDouble()
            filter_radius = self.mapper_settings["filter_radius"].GetDouble()
            if abs(filter_radius - penalty_filter_radius) > 1e-9:
                penalty_filter_settings = self.mapper_settings.Clone()
                penalty_filter_settings["filter_radius"].SetDouble(self.algorithm_settings["penalty_filter_radius"].GetDouble())
                self.penalty_filter = mapper_factory.CreateMapper(self.design_surface, self.design_surface, penalty_filter_settings)
                self.penalty_filter.Initialize()
            else:
                self.penalty_filter = self.mapper
        """

        self.data_logger = data_logger_factory.CreateDataLogger(self.model_part_controller, self.communicator, self.optimization_settings)
        self.data_logger.InitializeDataLogging()
        #Activate all the optimization_utilities.h in custum utilities
        self.optimization_utilities = KSO.OptimizationUtilities(self.design_surface, self.optimization_settings)

        #Bead optimization
        """
        # Identify fixed design areas
        KM.VariableUtils().SetFlag(KM.BOUNDARY, False, self.optimization_model_part.Nodes)

        radius = self.mapper_settings["filter_radius"].GetDouble()
        search_based_functions = KSO.SearchBasedFunctions(self.design_surface)

        for itr in range(self.algorithm_settings["fix_boundaries"].size()):
            sub_model_part_name = self.algorithm_settings["fix_boundaries"][itr].GetString()
            node_set = self.optimization_model_part.GetSubModelPart(sub_model_part_name).Nodes
            search_based_functions.FlagNodesInRadius(node_set, KM.BOUNDARY, radius)

        # Specify bounds and assign starting values for ALPHA
        if self.bead_side == "positive":
            KM.VariableUtils().SetScalarVar(KSO.ALPHA, 0.5, self.design_surface.Nodes, KM.BOUNDARY, False)
            self.lower_bound = 0.0
            self.upper_bound = 1.0
        elif self.bead_side == "negative":
            KM.VariableUtils().SetScalarVar(KSO.ALPHA, -0.5, self.design_surface.Nodes, KM.BOUNDARY, False)
            self.lower_bound = -1.0
            self.upper_bound = 0.0
        elif self.bead_side == "both":
            KM.VariableUtils().SetScalarVar(KSO.ALPHA, 0.0, self.design_surface.Nodes, KM.BOUNDARY, False)
            self.lower_bound = -1.0
            self.upper_bound = 1.0
        else:
            raise RuntimeError("Specified bead direction mode not supported!")

        # Initialize ALPHA_MAPPED according to initial ALPHA values
        self.mapper.Map(KSO.ALPHA, KSO.ALPHA_MAPPED)

        # Specify bead direction
        bead_direction = self.algorithm_settings["bead_direction"].GetVector()
        if len(bead_direction) == 0:
            self.model_part_controller.ComputeUnitSurfaceNormals()
            for node in self.design_surface.Nodes:
                normalized_normal = node.GetSolutionStepValue(KSO.NORMALIZED_SURFACE_NORMAL)
                node.SetValue(KSO.BEAD_DIRECTION,normalized_normal)

        elif len(bead_direction) == 3:
            norm = math.sqrt(bead_direction[0]**2 + bead_direction[1]**2 + bead_direction[2]**2)
            normalized_bead_direction = [value/norm for value in bead_direction]
            KM.VariableUtils().SetNonHistoricalVectorVar(KSO.BEAD_DIRECTION, normalized_bead_direction, self.design_surface.Nodes)
        else:
            raise RuntimeError("Wrong definition of bead direction. Options are: 1) [] -> takes surface normal, 2) [x.x,x.x,x.x] -> takes specified vector.")

        """
    # --------------------------------------------------------------------------
    def RunOptimizationLoop(self):
        timer = Timer()
        timer.StartTimer()
        total_iteration=0
        current_lambda_g=self.lambda_g_0
        current_lambda_h=self.lambda_h_0
        current_p_vect_ineq=self.p_vect_ineq_0
        current_p_vect_eq=self.p_vect_eq_0
        
        self.__IdentifyNumberInequalities()
       
        gamma=2.0
        for itr in range(self.number_ineq):
            current_lambda_g.append(0.0)
            current_p_vect_ineq.append(self.p)
        for itr in range(self.number_eq):
            current_lambda_h.append(0.0)
            current_p_vect_eq.append(self.p)

        g_gradient_vector_kratos=[]
        h_gradient_vector_kratos=[]
        
        is_design_converged = False
        is_max_total_iterations_reached = False
        previos_L = None

        for outer_iteration in range(1,self.max_outer_iterations+1):           
            for inner_iteration in range(1,self.max_inner_iterations+1):
                total_iteration += 1

                KM.Logger.Print("")
                KM.Logger.Print("===============================================================================")
                KM.Logger.PrintInfo("ShapeOpt", timer.GetTimeStamp(),  ": Starting iteration ",outer_iteration,".",inner_iteration,".",total_iteration,"(outer . inner. total)")
                KM.Logger.Print("===============================================================================\n")

                timer.StartNewLap()

                self.__InitializeNewShape(total_iteration)

                    
                #Begin of __analyzeShape(self)
                self.__AnalyzeShape(total_iteration)
                #Writing values of objectives and Gradient in variables of python.
                objective_value = self.communicator.getStandardizedValue(self.objectives[0]["identifier"].GetString())
                objGradientDict = self.communicator.getStandardizedGradient(self.objectives[0]["identifier"].GetString())
                WriteDictionaryDataOnNodalVariable(objGradientDict, self.optimization_model_part, KSO.DF1DX)

                #Writing values of constraints and gradient in variables of python, the constraints are standarized.
                constraint_vector=[]
                for constraint in self.constraints:
                    con_id = constraint["identifier"].GetString()
                    constraint_vector.append(self.communicator.getStandardizedValue(con_id))
                    conGradientDict = self.communicator.getStandardizedGradient(con_id)
                    gradient_variable = self.constraint_gradient_variables[con_id]["gradient"]
                    #Here we write the value of gradient on the DCi/DX
                    WriteDictionaryDataOnNodalVariable(conGradientDict, self.optimization_model_part, gradient_variable)    
                #End of __analyzeShape(self)

                #Begin of__computeShapeUpdate(self):
                self.mapper.Update()
                self.mapper.InverseMap(KSO.DF1DX, KSO.DF1DX_MAPPED) 
                #Here all the constraints are mapped
                for constraint in self.constraints:
                    con_id = constraint["identifier"].GetString()
                    gradient_contraint = self.constraint_gradient_variables[con_id]["gradient"]
                    mapped_gradient_variable = self.constraint_gradient_variables[con_id]["mapped_gradient"]
                    self.mapper.InverseMap(gradient_contraint, mapped_gradient_variable)
                    
                gp_utilities = self.optimization_utilities  
                g_values,g_gradient_variables,h_values,h_gradient_variables=self.__SeparateConstraints()
                    
                if self.number_ineq >0:
                    c_k=list()######Page 455
                    c_knext=list()
                    for itr in range(self.number_ineq):
                        c_k.append(max(g_values[itr],0))
                            
                if self.number_eq >0:
                    c_k_eq=list()######Page 455
                    c_knext_eq=list()
                    for itr in range(self.number_eq):
                        c_k_eq.append(max(h_values[itr],0))
                    
                    
                    #Definitivamente el metodo es general debido a que se involucran vectores de constraints. No nodal. 

                                
                KM.Logger.PrintInfo("ShapeOpt", "Assemble vector of objective gradient.")
                nabla_f = KM.Vector()
                gp_utilities.AssembleVector(nabla_f, KSO.DF1DX_MAPPED)
                    
                KM.Logger.PrintInfo("ShapeOpt", "Assemble vector of constraints gradient.")
                
                g_gradient_vector_kratos.clear()
                for itr in  range(len(g_gradient_variables)) :
                    g_gradient_vector_kratos.append( KM.Vector())
                    gp_utilities.AssembleVector(g_gradient_vector_kratos[itr], g_gradient_variables[itr])  
                
                h_gradient_vector_kratos.clear()
                for itr in  range(len(h_gradient_variables)):
                    h_gradient_vector_kratos.append( KM.Vector())
                    gp_utilities.AssembleVector(h_gradient_vector_kratos[itr], h_gradient_variables[itr])  
                
                conditions_ineq=0.0   
                for itr in range(len(g_values)):
                    if g_values[itr]>(-1*current_lambda_g[itr])/(2*current_p_vect_ineq[itr]):
                        conditions_ineq+=current_lambda_g[itr]*g_values[itr]+current_p_vect_ineq[itr]*g_values[itr]**2
                    else:
                        conditions_ineq+=(-1)*(current_lambda_g[itr])**2/(4*current_p_vect_ineq[itr])
                    
                conditions_eq=0.0
                for itr in range(len(h_values)):
                    conditions_eq+=current_lambda_h[itr]*h_values[itr]+current_p_vect_eq[itr]*h_values[itr]**2
                        
                    


                A=objective_value+conditions_ineq+conditions_eq
                    
                conditions_grad_ineq_vector=KM.Vector()
                conditions_grad_ineq_vector.Resize(nabla_f.Size())
                conditions_grad_ineq_vector.fill(0.0)

                conditions_grad_eq_vector=KM.Vector()
                conditions_grad_eq_vector.Resize(nabla_f.Size())
                conditions_grad_eq_vector.fill(0.0)
                                
                for itr in range(len(g_gradient_variables)):
                    if g_values[itr]>(-1*current_lambda_g[itr])/(2*current_p_vect_ineq[itr]):
                        conditions_grad_ineq_vector+=(current_lambda_g[itr]+2*current_p_vect_ineq[itr]*g_values[itr])*g_gradient_vector_kratos[itr]
                    else:
                        conditions_grad_ineq_vector+=conditions_grad_ineq_vector
                    
                for itr in range(len(h_gradient_variables)):
                    conditions_grad_eq_vector+=(current_lambda_h[itr]+2*current_p_vect_ineq[itr]*h_values[itr])*h_gradient_vector_kratos[itr]
                


                dA_dX_mapped=nabla_f+conditions_grad_ineq_vector+conditions_grad_eq_vector   
                search_direction_augmented=-1*dA_dX_mapped
                gp_utilities.AssignVectorToVariable(search_direction_augmented, KSO.SEARCH_DIRECTION)                
                self.optimization_utilities.ComputeControlPointUpdate(self.step_size)
                self.mapper.Map(KSO.CONTROL_POINT_UPDATE, KSO.SHAPE_UPDATE)
                
               
                # Log current optimization step and store values for next iteration
                additional_values_to_log = {}
                additional_values_to_log["step_size"] = self.algorithm_settings["line_search"]["step_size"].GetDouble()
                additional_values_to_log["outer_iteration"] = outer_iteration
                additional_values_to_log["inner_iteration"] = inner_iteration
                additional_values_to_log["lagrange_value"] = A
                #additional_values_to_log["lagrange_value_relative_change"] = dL_relative
                #additional_values_to_log["penalty_value"] = penalty_value
                additional_values_to_log["current_lambda_inequalities"] = current_lambda_g
                additional_values_to_log["current_lambda_equalities"] = current_lambda_h
                #additional_values_to_log["penalty_scaling"] = penalty_scaling
                additional_values_to_log["current_penalty_factor_inequalities"] = current_p_vect_ineq
                additional_values_to_log["current_penalty_factor_inequalities"] = current_p_vect_eq
                #additional_values_to_log["max_norm_objective_gradient"] = max_norm_objective_gradient

                self.data_logger.LogCurrentValues(total_iteration, additional_values_to_log)
                self.data_logger.LogCurrentDesign(total_iteration)

                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Time needed for current optimization step = ", timer.GetLapTime(), "s")
                KM.Logger.PrintInfo("ShapeOpt", "Time needed for total optimization so far = ", timer.GetTotalTime(), "s")

    	        # Convergence check of inner loop
                if total_iteration == self.max_total_iterations:
                    is_max_total_iterations_reached = True
                    break

                if inner_iteration >= self.min_inner_iterations and inner_iteration >1:
                    # In the first outer iteration, the constraint is not yet active and properly scaled. Therefore, the objective is used to check the relative improvement
                    if outer_iteration == 1:
                        if abs(self.data_logger.GetValues("rel_change_objective")[total_iteration]) < self.inner_iteration_tolerance:
                            break
                    else:
                        if abs(dL_relative) < self.inner_iteration_tolerance:
                            break

                #if penalty_value == 0.0:#change that with previous l
                #    is_design_converged = True
                #    break
            
             # Compute penalty factor such that estimated Lagrange multiplier is obtained
            
            #Update penalty factor vector
            if self.number_ineq >0:
                c_knext.clear()
                for i in range (self.number_ineq):
                    c_knext.append(max(g_values[i],0))
                                
                for i in range (len(c_knext)):
                    if  not abs(c_knext[i])<=((1/4)*abs(c_k[i])):
                        if current_p_vect_ineq[i]>=self.pmax:
                            current_p_vect_ineq[i]=self.pmax                        
                        else:
                            current_p_vect_ineq[i]=gamma*current_p_vect_ineq[i]
                c_k=c_knext.copy()     
            if self.number_eq >0:
                c_knext_eq.clear()
                for i in range(self.number_eq):    
                    c_knext_eq.append(max(h_values[i],0))
                
                for i in range (len(c_knext_eq)):
                   if  not abs(c_knext_eq[i])<=((1/4)*abs(c_k_eq[i])):
                        if current_p_vect_eq[i]>=self.pmax:
                            current_p_vect_eq[i]=self.pmax                        
                        else:
                            current_p_vect_eq[i]=gamma*current_p_vect_eq[i]                      
                c_k_eq=c_knext_eq.copy()


            # Update lambda
            
            for itr in range(len(g_values)):
                if g_values[itr]>(-1*current_lambda_g[itr])/(2*current_p_vect_ineq[itr]):
                    current_lambda_g[itr]=current_lambda_g[itr]+2*current_p_vect_ineq[itr]
                else:
                    current_lambda_g[itr]=0.0
            
            for itr in range(len(h_values)):
                current_lambda_h[itr]=current_lambda_h[itr]+2*current_p_vect_eq[itr]


            
            


            KM.Logger.Print("")
            KM.Logger.PrintInfo("ShapeOpt", "Time needed for current optimization step = ", timer.GetLapTime(), "s")
            KM.Logger.PrintInfo("ShapeOpt", "Time needed for total optimization so far = ", timer.GetTotalTime(), "s")

            # Check convergence of outer loop
            if outer_iteration == self.max_outer_iterations:
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Maximal outer iterations of optimization problem reached!")
                break

            if is_max_total_iterations_reached:
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Maximal total iterations of optimization problem reached!")
                break

            if is_max_total_iterations_reached:
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Maximal total iterations of optimization problem reached!")
                break

            if is_design_converged:
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Update of design variables is zero. Optimization converged!")
                break
    # --------------------------------------------------------------------------
    def FinalizeOptimizationLoop(self):
        self.analyzer.FinalizeAfterOptimizationLoop()
        self.data_logger.FinalizeDataLogging()

    # --------------------------------------------------------------------------
    def __isAlgorithmConverged(self):

        if self.opt_iteration > 1 :
            # Check if maximum iterations were reached
            if self.opt_iteration == self.algorithm_settings["max_iterations"].GetInt():
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Maximal iterations of optimization problem reached!")
                return True

            # Check for relative tolerance
            relative_change_of_objective_value = self.data_logger.GetValues("rel_change_objective")[self.opt_iteration]
            if abs(relative_change_of_objective_value) < self.algorithm_settings["relative_tolerance"].GetDouble():
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Optimization problem converged within a relative objective tolerance of ",self.algorithm_settings["relative_tolerance"].GetDouble(),"%.")
                return True

    # --------------------------------------------------------------------------
    def __InitializeNewShape(self,total_iteration):
        self.model_part_controller.UpdateTimeStep(total_iteration)

        """
        for node in self.design_surface.Nodes:
            new_shape_change = node.GetSolutionStepValue(KSO.ALPHA_MAPPED) * node.GetValue(KSO.BEAD_DIRECTION) * self.bead_height
            node.SetSolutionStepValue(KSO.SHAPE_CHANGE, new_shape_change)

        self.model_part_controller.DampNodalVariableIfSpecified(KSO.SHAPE_CHANGE)

        for node in self.design_surface.Nodes:
            shape_update = node.GetSolutionStepValue(KSO.SHAPE_CHANGE,0) - node.GetSolutionStepValue(KSO.SHAPE_CHANGE,1)
            node.SetSolutionStepValue(KSO.SHAPE_UPDATE, shape_update)
        """

        self.model_part_controller.UpdateMeshAccordingInputVariable(KSO.SHAPE_UPDATE)
        self.model_part_controller.SetReferenceMeshToMesh()

    # --------------------------------------------------------------------------
    def __AnalyzeShape(self,total_iteration):
        self.communicator.initializeCommunication()

        obj_id = self.objectives[0]["identifier"].GetString()
        self.communicator.requestValueOf(obj_id)
        self.communicator.requestGradientOf(obj_id)

        for constraint in self.constraints:
            con_id =  constraint["identifier"].GetString()
            self.communicator.requestValueOf(con_id)
            self.communicator.requestGradientOf(con_id)

        self.analyzer.AnalyzeDesignAndReportToCommunicator(self.optimization_model_part, total_iteration, self.communicator) #self.opt_iteration

    # --------------------------------------------------------------------------
    def __PostProcessGradientsObtainedFromAnalysis(self):
        # Compute surface normals if required
        if self.objectives[0]["project_gradient_on_surface_normals"].GetBool():
            self.model_part_controller.ComputeUnitSurfaceNormals()
        else:
            for itr in range(self.constraints.size()):
                if self.constraints[itr]["project_gradient_on_surface_normals"].GetBool():
                    self.model_part_controller.ComputeUnitSurfaceNormals()

        # Process objective gradients
        obj = self.objectives[0]
        obj_id = obj["identifier"].GetString()

        obj_gradients_dict = self.communicator.getStandardizedGradient(obj_id)

        nodal_variable = KM.KratosGlobals.GetVariable("DF1DX")
        WriteDictionaryDataOnNodalVariable(obj_gradients_dict, self.optimization_model_part, nodal_variable)

        # Projection on surface normals
        if obj["project_gradient_on_surface_normals"].GetBool():
            self.model_part_controller.ProjectNodalVariableOnUnitSurfaceNormals(nodal_variable)

        # Damping
        self.model_part_controller.DampNodalVariableIfSpecified(nodal_variable)

        # Mapping
        nodal_variable_mapped = KM.KratosGlobals.GetVariable("DF1DX_MAPPED")
        self.mapper.Update()
        self.mapper.InverseMap(nodal_variable, nodal_variable_mapped)
        self.mapper.Map(nodal_variable_mapped, nodal_variable_mapped)

        # Damping
        self.model_part_controller.DampNodalVariableIfSpecified(nodal_variable_mapped)

        # Process constraint gradients
        for itr in range(self.constraints.size()):
            con = self.constraints[itr]
            con_id = con["identifier"].GetString()

            eq_gradients_dict = self.communicator.getStandardizedGradient(con_id)

            nodal_variable = KM.KratosGlobals.GetVariable("DC"+str(itr+1)+"DX")
            WriteDictionaryDataOnNodalVariable(eq_gradients_dict, self.optimization_model_part, nodal_variable)

            # Projection on surface normals
            if con["project_gradient_on_surface_normals"].GetBool():
                self.model_part_controller.ProjectNodalVariableOnUnitSurfaceNormals(nodal_variable)

            # Damping
            self.model_part_controller.DampNodalVariableIfSpecified(nodal_variable)

            # Mapping
            nodal_variable_mapped = KM.KratosGlobals.GetVariable("DC"+str(itr+1)+"DX_MAPPED")
            self.mapper.InverseMap(nodal_variable, nodal_variable_mapped)
            self.mapper.Map(nodal_variable_mapped, nodal_variable_mapped)

            # Damping
            self.model_part_controller.DampNodalVariableIfSpecified(nodal_variable_mapped)

    # --------------------------------------------------------------------------
    def __ConvertAnalysisResultsToLengthDirectionFormat(self):
        # Convert objective results
        obj = self.objectives[0]
        obj_id = obj["identifier"].GetString()

        nodal_variable = KM.KratosGlobals.GetVariable("DF1DX")
        nodal_variable_mapped = KM.KratosGlobals.GetVariable("DF1DX_MAPPED")

        obj_value = self.communicator.getStandardizedValue(obj_id)
        obj_gradient = ReadNodalVariableToList(self.design_surface, nodal_variable)
        obj_gradient_mapped = ReadNodalVariableToList(self.design_surface, nodal_variable_mapped)

        dir_obj, len_obj = self.__ConvertToLengthDirectionFormat(obj_value, obj_gradient, obj_gradient_mapped)
        dir_obj = dir_obj

        # Convert constraints
        len_eqs = []
        dir_eqs = []
        len_ineqs = []
        dir_ineqs = []

        for itr in range(self.constraints.size()):
            con = self.constraints[itr]
            con_id = con["identifier"].GetString()

            nodal_variable = KM.KratosGlobals.GetVariable("DC"+str(itr+1)+"DX")
            nodal_variable_mapped = KM.KratosGlobals.GetVariable("DC"+str(itr+1)+"DX_MAPPED")

            value = self.communicator.getStandardizedValue(con_id)
            gradient = ReadNodalVariableToList(self.design_surface, nodal_variable)
            gradient_mapped = ReadNodalVariableToList(self.design_surface, nodal_variable_mapped)

            direction, length = self.__ConvertToLengthDirectionFormat(value, gradient, gradient_mapped)

            if con["type"].GetString()=="=":
                dir_eqs.append(direction)
                len_eqs.append(length)
            else:
                dir_ineqs.append(direction)
                len_ineqs.append(length)

        return len_obj, dir_obj, len_eqs, dir_eqs, len_ineqs, dir_ineqs

    # --------------------------------------------------------------------------
    @staticmethod
    def __ConvertToLengthDirectionFormat(value, gradient, modified_gradient):
        norm_inf = cm.NormInf3D(modified_gradient)
        if norm_inf > 1e-12:
            direction = cm.ScalarVectorProduct(-1/norm_inf,modified_gradient)
            length = -value/cm.Dot(gradient, direction)
        else:
            KM.Logger.PrintWarning("ShapeOpt::AlgorithmTrustRegion", "Vanishing norm-infinity for gradient detected!")
            direction = modified_gradient
            length = 0.0

        return direction, length

    # --------------------------------------------------------------------------
    def __DetermineMaxStepLength(self):
        if self.opt_iteration < 4:
            return self.algorithm_settings["max_step_length"].GetDouble()
        else:
            obj_id = self.objectives[0]["identifier"].GetString()
            current_obj_val = self.communicator.getStandardizedValue(obj_id)
            obj_history = self.data_logger.GetValues("response_value")[obj_id]
            step_history = self.data_logger.GetValues("step_length")

            # Check for osciallation
            objective_is_oscillating = False
            is_decrease_1 = (current_obj_val - obj_history[self.opt_iteration-1])< 0
            is_decrease_2 = (obj_history[self.opt_iteration-1] - obj_history[self.opt_iteration-2])<0
            is_decrease_3 = (current_obj_val - obj_history[self.opt_iteration-3])< 0
            if (is_decrease_1 and is_decrease_2== False and is_decrease_3) or (is_decrease_1== False and is_decrease_2 and is_decrease_3==False):
                objective_is_oscillating = True

            # Reduce step length if certain conditions are fullfilled
            if objective_is_oscillating:
                return step_history[self.opt_iteration-1]*self.algorithm_settings["step_length_reduction_factor"].GetDouble()
            else:
                return step_history[self.opt_iteration-1]

    # --------------------------------------------------------------------------
    @staticmethod
    def __ExpressInStepLengthUnit(len_obj, len_eqs, len_ineqs, step_length):
        len_bar_obj = 1/step_length * len_obj
        len_bar_eqs = cm.ScalarVectorProduct(1/step_length, len_eqs)
        len_bar_ineqs = cm.ScalarVectorProduct(1/step_length, len_ineqs)
        return len_bar_obj, len_bar_eqs, len_bar_ineqs

    # --------------------------------------------------------------------------
    def __DetermineStep(self, len_obj, dir_obj, len_eqs, dir_eqs, len_ineqs, dir_ineqs):
        KM.Logger.Print("")
        KM.Logger.PrintInfo("ShapeOpt", "Starting determination of step...")

        timer = Timer()
        timer.StartTimer()

        # Create projector object wich can do the projection in the orthogonalized subspace
        projector = Projector(len_obj, dir_obj, len_eqs, dir_eqs, len_ineqs, dir_ineqs, self.algorithm_settings)

        # 1. Test projection if there is room for objective improvement
        # I.e., the actual step length to become feasible for an inactive threshold is smaller than 1 and hence a part of the step can be dedicated to objective improvement
        len_obj_test = 0.01
        inactive_threshold = 100
        test_norm_dX, is_projection_sucessfull = projector.RunProjection(len_obj_test, inactive_threshold)

        KM.Logger.PrintInfo("ShapeOpt", "Time needed for one projection step = ", timer.GetTotalTime(), "s")

        # 2. Determine step following two different modes depending on the previos found step length to the feasible domain
        if is_projection_sucessfull:
            if test_norm_dX < 1: # Minimizing mode
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Computing projection case 1...")

                func = lambda len_obj: projector.RunProjection(len_obj, inactive_threshold)

                len_obj_min = len_obj_test
                len_obj_max = 1.3
                bi_target = 1
                bi_tolerance = self.algorithm_settings["bisectioning_tolerance"].GetDouble()
                bi_max_itr = self.algorithm_settings["bisectioning_max_itr"].GetInt()
                len_obj_result, bi_itrs, bi_err = cm.PerformBisectioning(func, len_obj_min, len_obj_max, bi_target, bi_tolerance, bi_max_itr)

                projection_results = projector.GetDetailedResultsOfLatestProjection()

            else: # Correction mode
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Computing projection case 2...")

                len_obj = self.algorithm_settings["obj_share_during_correction"].GetDouble()
                func = lambda threshold: projector.RunProjection(len_obj, threshold)

                threshold_min = 0
                threshold_max = 1.3
                bi_target = 1
                bi_tolerance = self.algorithm_settings["bisectioning_tolerance"].GetDouble()
                bi_max_itr = self.algorithm_settings["bisectioning_max_itr"].GetInt()
                l_threshold_result, bi_itrs, bi_err = cm.PerformBisectioning(func, threshold_min, threshold_max, bi_target, bi_tolerance, bi_max_itr)

                projection_results = projector.GetDetailedResultsOfLatestProjection()
        else:
            raise RuntimeError("Case of not converged test projection not yet implemented yet!")

        KM.Logger.Print("")
        KM.Logger.PrintInfo("ShapeOpt", "Time needed for determining step = ", timer.GetTotalTime(), "s")

        process_details = { "test_norm_dX": test_norm_dX,
                            "bi_itrs":bi_itrs,
                            "bi_err":bi_err,
                            "adj_len_obj": projection_results["adj_len_obj"],
                            "adj_len_eqs": projection_results["adj_len_eqs"],
                            "adj_len_ineqs": projection_results["adj_len_ineqs"] }

        return projection_results["dX"], process_details

    # --------------------------------------------------------------------------
    def __ComputeShapeUpdate(self, dX_bar, step_length):
        # Compute update in regular units
        dX = cm.ScalarVectorProduct(step_length,dX_bar)

        WriteListToNodalVariable(dX, self.design_surface, KSO.SHAPE_UPDATE)
        self.optimization_utilities.AddFirstVariableToSecondVariable(KSO.SHAPE_UPDATE, KSO.SHAPE_CHANGE)

        return dX

    # --------------------------------------------------------------------------
    def __LogCurrentOptimizationStep(self, additional_values_to_log):
        self.data_logger.LogCurrentValues(self.opt_iteration, additional_values_to_log)
        self.data_logger.LogCurrentDesign(self.opt_iteration)

    # --------------------------------------------------------------------------
    def __CombineConstraintDataToOrderedList(self, eqs_data_list, ineqs_data_list):
        num_eqs = 0
        num_ineqs = 0
        combined_list = []

        # Order is given by appearance of constraints in optimization settings
        for itr in range(self.constraints.size()):
            if self.constraints[itr]["type"].GetString()=="=":
                combined_list.append(eqs_data_list[num_eqs])
                num_eqs = num_eqs+1
            else:
                combined_list.append(ineqs_data_list[num_ineqs])
                num_ineqs = num_ineqs+1

        return combined_list
    def __SeparateConstraints(self):
        equality_constraint_values = []
        equality_constraint_gradient = []
        inequality_constraint_values = []
        inequality_constraint_gradient = []
      
        for constraint in self.constraints:
            identifier = constraint["identifier"].GetString()
            if self.__InequalityConstraint(constraint)==True:
                constraint_value = self.communicator.getStandardizedValue(identifier)
                inequality_constraint_values.append(constraint_value)
                inequality_constraint_gradient.append(
                    self.constraint_gradient_variables[identifier]["mapped_gradient"])
            else:
                constraint_value = self.communicator.getStandardizedValue(identifier)
                equality_constraint_values.append(constraint_value)
                equality_constraint_gradient.append(
                    self.constraint_gradient_variables[identifier]["mapped_gradient"])



        return inequality_constraint_values,inequality_constraint_gradient, equality_constraint_values,equality_constraint_gradient
    

    def __IdentifyNumberInequalities(self):
        for constraint in self.constraints:
            identifier = constraint["identifier"].GetString()
            if self.__InequalityConstraint(constraint)==True:
               self.number_ineq+=1
            else:
                self.number_eq+=1



        

    # --------------------------------------------------------------------------
    def __InequalityConstraint(self, constraint):
        
        if constraint["type"].GetString() != "=":
            return True
        else:
            return False
# ==============================================================================
class Projector():
    # --------------------------------------------------------------------------
    def __init__(self, len_obj, dir_obj, len_eqs, dir_eqs, len_ineqs, dir_ineqs, settings):

        # Store settings
        self.far_away_length = settings["far_away_length"].GetDouble()
        self.subopt_max_itr = settings["subopt_max_itr"].GetInt()
        self.subopt_tolerance = settings["subopt_tolerance"].GetDouble()

        # Initialize projection results
        self.are_projection_restuls_stored = False
        self.projection_results = {}

        # Reduce input data to relevant info
        self.input_len_obj = len_obj
        self.input_len_eqs = len_eqs
        self.input_len_ineqs = len_ineqs

        len_eqs, dir_eqs, remaining_eqs_entries = self.__ReduceToRelevantEqualityConstraints(len_eqs, dir_eqs)
        len_ineqs, dir_ineqs, remaining_ineqs_entries = self.__ReduceToRelevantInequalityConstraints(len_ineqs, dir_ineqs)

        # Store some working variables depening on data reduction
        self.remaining_eqs_entries = remaining_eqs_entries
        self.remaining_ineqs_entries = remaining_ineqs_entries
        self.len_eqs = len_eqs
        self.len_ineqs = len_ineqs
        self.num_eqs = len(len_eqs)
        self.num_ineqs = len(len_ineqs)
        self.num_unknowns = 1 + self.num_eqs + self.num_ineqs

        # Create orthogonal basis
        vector_space = [dir_obj]
        for itr in range(len(dir_eqs)):
            vector_space = cm.HorzCat(vector_space, dir_eqs[itr])
        for itr in range(len(dir_ineqs)):
            vector_space = cm.HorzCat(vector_space, dir_ineqs[itr])
        self.ortho_basis = cm.PerformGramSchmidtOrthogonalization(vector_space)

        # Transform directions to orthogonal space since they don't change with different projections
        self.dir_obj_o = cm.TranslateToNewBasis(dir_obj, self.ortho_basis)
        self.dir_eqs_o = cm.TranslateToNewBasis(dir_eqs, self.ortho_basis)
        self.dir_ineqs_o = cm.TranslateToNewBasis(dir_ineqs, self.ortho_basis)

        # Make sure directions of constraints are stored as matrix
        self.dir_eqs_o = cm.SafeConvertVectorToMatrix(self.dir_eqs_o)
        self.dir_ineqs_o = cm.SafeConvertVectorToMatrix(self.dir_ineqs_o)

    # --------------------------------------------------------------------------
    def RunProjection(self, len_obj, threshold):

        # Adjust halfspaces according input
        adj_len_obj, adj_len_eqs, adj_len_ineqs = self.__AdjustHalfSpacesAndHyperplanes(len_obj, threshold)

        # Determine position of border of halfspaces and hyperplanes
        pos_obj_o, pos_eqs_o, pos_ineqs_o = self.__DetermineConstraintBorders(adj_len_obj, adj_len_eqs, adj_len_ineqs)

        # Project current position onto intersection of
        current_position = cm.ZeroVector(self.num_unknowns)
        dlambda_hp = self.__ProjectToHyperplanes(current_position, self.dir_eqs_o, pos_eqs_o)

        # Project position and direction of halfspaces onto intersection of hyperplanes
        zero_position_eqs_o = cm.ZeroMatrix(self.num_unknowns,self.num_eqs)

        pos_obj_hp = self.__ProjectToHyperplanes(pos_obj_o, cm.HorzCat(self.dir_eqs_o, self.dir_obj_o), cm.HorzCat(pos_eqs_o, pos_obj_o))
        dir_obj_hp = self.__ProjectToHyperplanes(self.dir_obj_o, self.dir_eqs_o, zero_position_eqs_o)

        pos_ineqs_hp = []
        dir_ineqs_hp = []
        for itr in range(self.num_ineqs):
            pos_ineqs_hp_i = self.__ProjectToHyperplanes(pos_ineqs_o[itr], cm.HorzCat(self.dir_eqs_o, self.dir_ineqs_o[itr]), cm.HorzCat(pos_eqs_o, pos_ineqs_o[itr]))
            dir_ineqs_hp_i = self.__ProjectToHyperplanes(self.dir_ineqs_o[itr], self.dir_eqs_o, zero_position_eqs_o)

            pos_ineqs_hp.append(pos_ineqs_hp_i)
            dir_ineqs_hp.append(dir_ineqs_hp_i)

        # Project onto adjusted halfspaces along the intersection of hyperplanes
        dX_o, _, _, exit_code = self.__ProjectToHalfSpaces(dlambda_hp, cm.HorzCat(pos_ineqs_hp, pos_obj_hp), cm.HorzCat(dir_ineqs_hp, dir_obj_hp))

        # Determine return values
        if exit_code == 0:
            is_projection_sucessfull = True

            # Backtransformation and multiplication with -1 because the direction vectors are chosen opposite to the gradients such that the lengths are positive if violated
            dX = cm.ScalarVectorProduct(-1, cm.TranslateToOriginalBasis(dX_o, self.ortho_basis))
            norm_dX = cm.NormInf3D(dX)
        else:
            is_projection_sucessfull = False

            dX = []
            norm_dX = 1e10

        self.__StoreProjectionResults(norm_dX, dX, is_projection_sucessfull, adj_len_obj, adj_len_eqs, adj_len_ineqs)

        return norm_dX, is_projection_sucessfull


    # --------------------------------------------------------------------------
    def GetDetailedResultsOfLatestProjection(self):
        if self.are_projection_restuls_stored == False:
            raise RuntimeError("Projector::__StoreProjectionResults: No projection results stored yet!")

        return self.projection_results

    # --------------------------------------------------------------------------
    @staticmethod
    def __ReduceToRelevantEqualityConstraints(len_eqs, dir_eqs):
        len_eqs_relevant = []
        dir_eqs_relevant = []
        remaining_entries = []

        for itr in range(len(dir_eqs)):
            len_i = len_eqs[itr]
            dir_i = dir_eqs[itr]

            is_no_gradient_info_available = cm.NormInf3D(dir_i) < 1e-13

            if is_no_gradient_info_available:
                pass
            else:
                remaining_entries.append(itr)
                len_eqs_relevant.append(len_i)
                dir_eqs_relevant.append(dir_i)

        return len_eqs_relevant, dir_eqs_relevant, remaining_entries

    # --------------------------------------------------------------------------
    def __ReduceToRelevantInequalityConstraints(self, len_ineqs, dir_ineqs):
        len_ineqs_relevant = []
        dir_ineqs_relevant = []
        remaining_entries = []

        for itr in range(len(dir_ineqs)):
            len_i = len_ineqs[itr]
            dir_i = dir_ineqs[itr]

            is_no_gradient_info_available = cm.NormInf3D(dir_i) < 1e-13
            is_constraint_inactive_and_far_away = len_i < -self.far_away_length

            if is_no_gradient_info_available or is_constraint_inactive_and_far_away:
                pass
            else:
                remaining_entries.append(itr)
                len_ineqs_relevant.append(len_i)
                dir_ineqs_relevant.append(dir_i)

        return len_ineqs_relevant, dir_ineqs_relevant, remaining_entries

    # --------------------------------------------------------------------------
    def __AdjustHalfSpacesAndHyperplanes(self, len_obj, threshold):
        if threshold<len_obj:
            len_obj = threshold

        len_eqs = copy.deepcopy(self.len_eqs)
        len_ineqs = copy.deepcopy(self.len_ineqs)

        for itr in range(self.num_eqs):
            len_eqs[itr] = min(max(self.len_eqs[itr],-threshold),threshold)

        for itr in range(self.num_ineqs):
            len_ineqs[itr] = min(self.len_ineqs[itr],threshold)

        return len_obj, len_eqs, len_ineqs

    # --------------------------------------------------------------------------
    def __DetermineConstraintBorders(self, len_obj, len_eqs, len_ineqs):
        pos_obj = cm.ScalarVectorProduct(-len_obj,self.dir_obj_o)

        pos_eqs = []
        pos_ineqs = []
        for i in range(self.num_eqs):
            pos_eqs.append(cm.ScalarVectorProduct(-len_eqs[i],self.dir_eqs_o[i]))

        for i in range(self.num_ineqs):
            pos_ineqs.append(cm.ScalarVectorProduct(-len_ineqs[i],self.dir_ineqs_o[i]))

        return pos_obj, pos_eqs, pos_ineqs

    # --------------------------------------------------------------------------
    @staticmethod
    def __ProjectToHyperplanes(vector, dir_hps, pos_hps):
        if cm.IsEmpty(dir_hps):
            return vector

        num_hps = len(dir_hps)

        tmp_mat = cm.Prod(cm.Trans(dir_hps),dir_hps)
        tmp_vec = [ cm.Dot(dir_hps[j],cm.Minus(pos_hps[j],vector)) for j in range(num_hps) ]

        tmp_solution = cm.SolveLinearSystem(tmp_mat,tmp_vec)

        return cm.Plus(cm.Prod(dir_hps,tmp_solution),vector)

    # --------------------------------------------------------------------------
    def __ProjectToHalfSpaces(self, dX0, pos_hss, dir_hss):
        A = cm.Trans(dir_hss)
        b = [ cm.Dot(pos_hss[i],dir_hss[i]) for i in range(cm.RowSize(A)) ]

        dX_o, subopt_itr, error, exit_code = cm.QuadProg(A, b, self.subopt_max_itr, self.subopt_tolerance)

        # Consider initial delta
        dX_o = cm.Plus(dX_o,dX0)

        return dX_o, subopt_itr, error, exit_code

    # --------------------------------------------------------------------------
    def __StoreProjectionResults(self, norm_dX, dX, is_projection_sucessfull, adj_len_obj, adj_len_eqs, adj_len_ineqs):
        self.are_projection_restuls_stored = True
        self.projection_results["norm_dX"] = norm_dX
        self.projection_results["dX"] = dX
        self.projection_results["is_projection_sucessfull"] = is_projection_sucessfull
        self.projection_results["adj_len_obj"] = adj_len_obj
        self.projection_results["adj_len_eqs"], self.projection_results["adj_len_ineqs"] = self.__CompleteConstraintLengthsWithRemovedEntries(adj_len_eqs, adj_len_ineqs)

    # --------------------------------------------------------------------------
    def __CompleteConstraintLengthsWithRemovedEntries(self, len_eqs, len_ineqs):
        # Complete list of eqs
        complete_list_eqs = copy.deepcopy(self.input_len_eqs)
        for itr in range(len(len_eqs)):
            original_eq_number = self.remaining_eqs_entries[itr]
            complete_list_eqs[original_eq_number] = len_eqs[itr]

        # Complete list of ineqs
        complete_list_ineqs = copy.deepcopy(self.input_len_ineqs)
        for itr in range(len(len_ineqs)):
            original_eq_number = self.remaining_ineqs_entries[itr]
            complete_list_ineqs[original_eq_number] = len_ineqs[itr]

        return complete_list_eqs, complete_list_ineqs

# ==============================================================================
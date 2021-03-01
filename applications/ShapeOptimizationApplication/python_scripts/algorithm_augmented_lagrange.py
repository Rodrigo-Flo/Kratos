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

from .custom_timer import Timer
from .custom_variable_utilities import WriteDictionaryDataOnNodalVariable, ReadNodalVariableToList, WriteListToNodalVariable


# ==============================================================================
class AlgorithmAugmentedLagrange(OptimizationAlgorithm):

    def __init__(self, optimization_settings, analyzer, communicator, model_part_controller):
        default_algorithm_settings = KM.Parameters("""
        {
            "name"                    : "augmented_lagrange",
            "max_correction_share"    : 0.75,
            "max_total_iterations"    : 500,
            "max_outer_iterations"    : 100,
            "max_inner_iterations"    : 50,
            "min_inner_iterations"    :  3,
            "gamma"                   :  2.0,
            "penalty_factor_initial"  : 1.0,
            "inner_iteration_tolerance"      : 1e-0,
            "line_search" : {
                "line_search_type"           : "manual_stepping",
                "normalize_search_direction" : true,
                "step_size"                  : 1.0,
                "estimation_tolerance"       : 0.1,
                "increase_factor"            : 1.25,
                "max_increase_factor"        : 3.0
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
        self.A=0.0
        self.dA_relative=0.0
        self.lambda_g_0=[]
        self.lambda_h_0=[]
        self.p_vect_ineq_0=[]
        self.p_vect_eq_0=[]
        self.number_ineq=0
        self.number_eq=0
        self.p= self.algorithm_settings["penalty_factor_initial"].GetDouble()#1.0#1387281818.0#1e9
        self.pmax=1e+10*self.p
        self.gamma=self.algorithm_settings["gamma"].GetDouble()#10.0

        self.constraint_gradient_variables = {}
        for itr, constraint in enumerate(self.constraints):
            self.constraint_gradient_variables.update({
                constraint["identifier"].GetString() : {
                    "gradient": KM.KratosGlobals.GetVariable("DC"+str(itr+1)+"DX"),
                    "mapped_gradient": KM.KratosGlobals.GetVariable("DC"+str(itr+1)+"DX_MAPPED")
                }
            })

        
        self.estimation_tolerance = self.algorithm_settings["line_search"]["estimation_tolerance"].GetDouble()
        self.step_size = self.algorithm_settings["line_search"]["step_size"].GetDouble()
        self.line_search_type = self.algorithm_settings["line_search"]["line_search_type"].GetString()
        self.increase_factor = self.algorithm_settings["line_search"]["increase_factor"].GetDouble()
        self.max_step_size = self.algorithm_settings["line_search"]["step_size"].GetDouble()*self.algorithm_settings["line_search"]["max_increase_factor"].GetDouble()
      
        self.max_total_iterations = self.algorithm_settings["max_total_iterations"].GetInt()
        self.max_outer_iterations = self.algorithm_settings["max_outer_iterations"].GetInt()
        
        self.max_inner_iterations = self.algorithm_settings["max_inner_iterations"].GetInt()
        self.min_inner_iterations = self.algorithm_settings["min_inner_iterations"].GetInt()
        self.inner_iteration_tolerance = self.algorithm_settings["inner_iteration_tolerance"].GetDouble()

        self.optimization_model_part = model_part_controller.GetOptimizationModelPart()
        
        #Variables for steepest descendent
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.SEARCH_DIRECTION)
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.DADX_MAPPED)
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.DIFF_)
        self.optimization_model_part.AddNodalSolutionStepVariable(KSO.H_MATRIX)

# ==============================================================================
    def CheckApplicability(self):
        if self.objectives.size() > 1:
            raise RuntimeError("Augmented Lagrange Method algorithm only supports one objective function at the moment!")
# ==============================================================================
    def InitializeOptimizationLoop(self):
        self.model_part_controller.Initialize()
        self.model_part_controller.SetMinimalBufferSize(2)        
        self.optimization_model_part.SetBufferSize(2)
    
        
        self.analyzer.InitializeBeforeOptimizationLoop()
        self.design_surface = self.model_part_controller.GetDesignSurface()
        
        

        self.mapper = mapper_factory.CreateMapper(self.design_surface, self.design_surface, self.mapper_settings)
        self.mapper.Initialize()
    

        self.data_logger = data_logger_factory.CreateDataLogger(self.model_part_controller, self.communicator, self.optimization_settings)
        self.data_logger.InitializeDataLogging()
        #Activate all the optimization_utilities.h in custum utilities
        self.optimization_utilities = KSO.OptimizationUtilities(self.design_surface, self.optimization_settings)

       
# ==============================================================================
    def RunOptimizationLoop(self):
        timer = Timer()
        timer.StartTimer()
        total_iteration=0
        
        current_lambda_g=self.lambda_g_0
        current_lambda_h=self.lambda_h_0
        current_p_vect_ineq=self.p_vect_ineq_0
        current_p_vect_eq=self.p_vect_eq_0
        
        self.__IdentifyNumberInequalities()
        current_lambda_g,current_p_vect_ineq,current_lambda_h,current_p_vect_eq=self.__InitializeLagrangeMultipliersAndPenalties(current_lambda_g,current_p_vect_ineq,current_lambda_h,current_p_vect_eq)
        
        g_gradient_vector_kratos=[]
        h_gradient_vector_kratos=[]
        scale_g_vector=[]
        scale_h_vector=[]
        
        is_design_converged = False
        is_max_total_iterations_reached = False
        outer_iteration=1
        n1_outer_absolute=0
        n2_outer_relative=0
        while is_design_converged==False: 
            #for outer_iteration in range(1,self.max_outer_iterations+1):           
            n1=0
            for inner_iteration in range(1,self.max_inner_iterations+1):
                total_iteration += 1

                KM.Logger.Print("")
                KM.Logger.Print("===============================================================================")
                KM.Logger.PrintInfo("ShapeOpt", timer.GetTimeStamp(),  ": Starting iteration ",outer_iteration,".",inner_iteration,".",total_iteration,"(outer . inner. total)")
                KM.Logger.Print("===============================================================================\n")
                timer.StartNewLap()

                self.__InitializeNewShape(total_iteration)
            
                self.__AnalyzeShape(total_iteration)

                self.__Mapping()
                gp_utilities = self.optimization_utilities
                
                objective_value = self.communicator.getStandardizedValue(self.objectives[0]["identifier"].GetString())      
                g_values,g_gradient_variables,h_values,h_gradient_variables=self.__SeparateConstraints()

                """
                if (total_iteration==2):
                    current_p_vect_ineq,current_p_vect_eq=self.__Scale_Penalties(objective_value,g_values,h_values)
                """
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

            #-------------Scaling constraints----------------------------------------------------------------------#
                if(inner_iteration==1): 
                    scale_g_vector.clear()
                    scale_h_vector.clear()
                for itr in range(len(g_gradient_variables)):
                    g_values[itr]=self.__Scale_constraints(inner_iteration,itr,
                                        scale_g_vector,g_values,nabla_f,g_gradient_vector_kratos)  
                for itr in range(len(h_gradient_variables)):
                    h_values[itr]=self.__Scale_constraints(inner_iteration,itr,
                                        scale_h_vector,h_values,nabla_f,h_gradient_vector_kratos)
            #---------------End of scaling------------------------------------------------------------------------#
               
                if (inner_iteration==2 and outer_iteration==1):
                    c_k,c_knext,c_k_eq,c_knext_eq=self.__AreInitialConstraintFeasible(g_values,h_values)
            #----------------------------------------------------------------------------------------------------#
                
                if (inner_iteration==1 and outer_iteration>1):  
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
                                    current_p_vect_ineq[i]=self.gamma*current_p_vect_ineq[i]
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
                                    current_p_vect_eq[i]=self.gamma*current_p_vect_eq[i]                      
                        c_k_eq=c_knext_eq.copy()    

                    # Update lambda
                    for itr in range(len(g_values)):
                        if g_values[itr]>(-1*current_lambda_g[itr])/(2*current_p_vect_ineq[itr]):
                            current_lambda_g[itr]=current_lambda_g[itr]+2*current_p_vect_ineq[itr]*g_values[itr]
                        else:
                            current_lambda_g[itr]=0.0
                
                    for itr in range(len(h_values)):
                        current_lambda_h[itr]=current_lambda_h[itr]+2*current_p_vect_eq[itr]*h_values[itr]     

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
                self.A=A#Used by the AdjustStepSize Function
                """
                if self.line_search_type == "adaptive_stepping" and inner_iteration > 1:
                   self.__AdjustStepSize(previous_A)
                elif inner_iteration==1:
                    self.step_size=self.algorithm_settings["line_search"]["step_size"].GetDouble()
                """
                
                
                if total_iteration==1:
                    dA_relative = 0.0
                else:
                    dA_relative = 100*(1-(previous_A/A)) 
                self.dA_relative=dA_relative#Used by the Log Information

                if inner_iteration ==1:
                    if total_iteration>1:
                        is_design_converged,n1_outer_absolute,n2_outer_relative=self.__CheckConvergence(A,A_init_inner,outer_iteration,n1_outer_absolute,n2_outer_relative,
                        0.0001,self.inner_iteration_tolerance,self.max_outer_iterations,"outer",n_iteration_1=1,n_iteration_2=2,)
                        if is_design_converged: 
                            KM.Logger.Print("")
                            self.__LogCurrentOptimizationStep(outer_iteration,inner_iteration,
                                                current_lambda_g,current_lambda_h,current_p_vect_ineq,current_p_vect_eq,
                                                total_iteration)
                            break
                    A_init_inner=A
                                

                conditions_grad_ineq_vector=KM.Vector()
                conditions_grad_ineq_vector.Resize(nabla_f.Size())
                conditions_grad_ineq_vector.fill(0.0)

                conditions_grad_eq_vector=KM.Vector()
                conditions_grad_eq_vector.Resize(nabla_f.Size())
                conditions_grad_eq_vector.fill(0.0)

                                               
                for itr in range(len(g_gradient_variables)):
                    if g_values[itr]>(-1*current_lambda_g[itr])/(2*current_p_vect_ineq[itr]):
                        conditions_grad_ineq_vector+=(current_lambda_g[itr]*scale_g_vector[itr]+2*current_p_vect_ineq[itr]*g_values[itr])*g_gradient_vector_kratos[itr]
                    else:
                        conditions_grad_ineq_vector+=conditions_grad_ineq_vector
                    
                for itr in range(len(h_gradient_variables)):
                    conditions_grad_eq_vector+=(current_lambda_h[itr]*scale_h_vector[itr]+2*current_p_vect_eq[itr]*h_values[itr])*h_gradient_vector_kratos[itr]#(current_lambda_h[itr]+2*current_p_vect_eq[itr]*h_values[itr])*h_gradient_vector_kratos[itr]
                
                dA_dX_mapped=nabla_f+conditions_grad_ineq_vector+conditions_grad_eq_vector   
                gp_utilities.AssignVectorToVariable(dA_dX_mapped,KSO.DADX_MAPPED)
                #search_direction_augmented=-1*dA_dX_mapped


                
                    

                 #---Quasi_Newton_Method-----#
                if inner_iteration==1:
                    search_direction_augmented=-1*dA_dX_mapped#H_.__mul__(dA_dX_mapped) 
                    gp_utilities.AssignVectorToVariable(search_direction_augmented, KSO.SEARCH_DIRECTION)
                    
                    I=KM.Matrix()
                    I.Resize(3,3)
                    I.fill_identity()
                    H_=KM.Matrix()
                    H_.Resize(3,3)
                    H_.fill_identity()
                    for node in self.design_surface.Nodes:
                        node.SetSolutionStepValue(KSO.H_MATRIX,H_)
                
                else:
                    y_=dA_dX_mapped-dA_dX_mapped_previous
                    gp_utilities.AssignVectorToVariable(y_, KSO.DIFF_)
                    #if abs(dA_relative) > 0.1:
                    #    print("QUASI_NEWTON_USED----------------------")
                    for node in self.design_surface.Nodes: 
                        s_nodal=node.GetSolutionStepValue(KSO.CONTROL_POINT_UPDATE)
                        y_nodal=node.GetSolutionStepValue(KSO.DIFF_)
                        phi_grad=node.GetSolutionStepValue(KSO.DADX_MAPPED)
                        H_=node.GetSolutionStepValue(KSO.H_MATRIX)
                        s_matrix=KM.Matrix()
                        y_matrix= KM.Matrix()
                        gp_utilities.AssembleMatrix_Column(s_matrix,s_nodal)
                        gp_utilities.AssembleMatrix_Column(y_matrix,y_nodal)
                        sigma=(s_matrix.transpose().__mul__(y_matrix))[0,0]
                        Left= I-(s_matrix.__mul__(y_matrix.transpose()))*(1/sigma)
                        Right=I-(y_matrix.__mul__(s_matrix.transpose()))*(1/sigma)
                        Add_term=(s_matrix.__mul__(s_matrix.transpose()))*(1/sigma)
                        PreMul=Left.__mul__(H_)
                        PosMul=PreMul.__mul__(Right)
                        H_=PosMul.__add__(Add_term)
                        node.SetSolutionStepValue(KSO.H_MATRIX,H_)
                        search_direction_nodal=(H_.__mul__(phi_grad)).__mul__(-1)
                        node.SetSolutionStepValue(KSO.SEARCH_DIRECTION,search_direction_nodal)
                    #else:   
                    #    for node in self.design_surface.Nodes:
                    #        phi_grad=node.GetSolutionStepValue(KSO.DADX_MAPPED)
                    #        H_=KM.Matrix()
                    #        H_.Resize(3,3)
                    #        H_.fill_identity()
                    #        node.SetSolutionStepValue(KSO.H_MATRIX,H_)
                    #        search_direction_nodal=-1*phi_grad
                    #        node.SetSolutionStepValue(KSO.SEARCH_DIRECTION,search_direction_nodal)



                #if inner_iteration==1:#total_iteration>1
                #    self.step_size=self.algorithm_settings["line_search"]["step_size"].GetDouble()
                
                #else:
                if total_iteration>1:
                    self.step_size=self.algorithm_settings["line_search"]["step_size"].GetDouble()#self.step_size=gp_utilities.CalculateStepSize_BB(dA_dX_mapped,dA_dX_mapped_previous,self.step_size)    
                #if total_iteration>1:
                #    
                #    if self.step_size<self.max_step_size*0.25:
                #        self.step_size=self.max_step_size*0.8
                #else:
                #    self.step_size=self.algorithm_settings["line_search"]["step_size"].GetDouble()
                
                
                self.optimization_utilities.ComputeControlPointUpdate(1/self.step_size)
                

                    
                dA_dX_mapped_previous=dA_dX_mapped
                self.mapper.Map(KSO.CONTROL_POINT_UPDATE, KSO.SHAPE_UPDATE)
                self.model_part_controller.DampNodalVariableIfSpecified(KSO.SHAPE_UPDATE)
                
                self.__LogCurrentOptimizationStep(outer_iteration,inner_iteration,
                                                current_lambda_g,current_lambda_h,current_p_vect_ineq,current_p_vect_eq,
                                                total_iteration)
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Time needed for current optimization step = ", timer.GetLapTime(), "s")
                KM.Logger.PrintInfo("ShapeOpt", "Time needed for total optimization so far = ", timer.GetTotalTime(), "s")
                
                previous_A=A
    	        # Convergence check of inner loop
                
                if total_iteration == self.max_total_iterations:
                    is_max_total_iterations_reached = True
                    break
                if inner_iteration >1:            
                    if abs(dA_relative) < self.inner_iteration_tolerance:
                        n1+=1
                    if n1==2:
                        break

            outer_iteration+=1
            KM.Logger.Print("")
            KM.Logger.PrintInfo("ShapeOpt", "Time needed for current optimization step = ", timer.GetLapTime(), "s")
            KM.Logger.PrintInfo("ShapeOpt", "Time needed for total optimization so far = ", timer.GetTotalTime(), "s")

            if is_max_total_iterations_reached:
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Maximal total iterations of optimization problem reached!")
                break
            if is_design_converged:
                break
       
# ==============================================================================
    def FinalizeOptimizationLoop(self):
        self.analyzer.FinalizeAfterOptimizationLoop()
        self.data_logger.FinalizeDataLogging()
# ==============================================================================
    def __CheckConvergence(self,F_knext,F_k,current_iteration,n1,n2,
                    absolute_tolerance,relative_tolerance,max_iterations,type_iteration,n_iteration_1=1,n_iteration_2=1,):
        
        n1=n1
        n2=n2
        if current_iteration == max_iterations:
            flag=True
            KM.Logger.Print("")
            KM.Logger.PrintInfo("ShapeOpt", "Maximal", type_iteration, "iterations of optimization problem reached!")
            return flag,n1,n2
        else:        
            dF1=abs(F_knext-F_k)
        
        if dF1 >absolute_tolerance:
            n1=0
        else:
            n1+=1
        if n1>=n_iteration_1:
            flag=True
            KM.Logger.PrintInfo("ShapeOpt","Optimization problem converged within a absolut tolerance of",
            dF1,"%."," for the Augmented Lagrange Function")
            return flag,n1,n2
        else:
            dF2=100*dF1/(max(abs(F_knext),10e-6))
        if dF2>relative_tolerance:
            n2=0
        else:
            n2+=1
        if n2>=n_iteration_2:
            flag=True
            KM.Logger.PrintInfo("ShapeOpt","Optimization problem converged within a relative tolerance of",
            dF2,"%."," for the Augmented Lagrange Function")
            return flag,n1,n2
        else:
            flag=False
            return flag,n1,n2
# ==============================================================================
    def __InitializeNewShape(self,total_iteration):
        self.model_part_controller.UpdateTimeStep(total_iteration)
        self.model_part_controller.UpdateMeshAccordingInputVariable(KSO.SHAPE_UPDATE)
        self.model_part_controller.SetReferenceMeshToMesh()
# ==============================================================================
    def __InitializeLagrangeMultipliersAndPenalties(self,current_lambda_g,current_p_vect_ineq,current_lambda_h,current_p_vect_eq):
        for itr in range(self.number_ineq):
            current_lambda_g.append(0.0)
            current_p_vect_ineq.append(self.p)
        for itr in range(self.number_eq):
            current_lambda_h.append(0.0)
            current_p_vect_eq.append(self.p)
        return current_lambda_g,current_p_vect_ineq,current_lambda_h,current_p_vect_eq
# ==============================================================================
    def __Scale_Penalties(self,objective_value,g_values,h_values):
        current_p_vect_ineq=[]
        current_p_vect_eq=[]
        p=0.0
        for itr in range(self.number_ineq):
            p+=(max(0,g_values[itr]))**2
        for itr in range(self.number_eq):
            p+=(h_values[itr])**2
        
        if p>0.0:
            for itr in range(self.number_ineq):
                current_p_vect_ineq.append(abs(objective_value)/p)
        else:
            for itr in range(self.number_ineq):
                current_p_vect_ineq.append(1.0)

        if p>0.0:
            for itr in range(self.number_eq):
                current_p_vect_eq.append(abs(objective_value)/p)
        else:
            for itr in range(self.number_eq):
                current_p_vect_eq.append(1.0)

        return current_p_vect_ineq,current_p_vect_eq
# ==============================================================================
    def  __Scale_constraints(self,initial_iteration,current_iteration,scale_vector,constraint_value,objective_gradient_vector,constraint_gradient_vector):
        if(initial_iteration==1):
            #scale_vector.clear()
            if constraint_gradient_vector[current_iteration].norm_inf()!=0.0:
                scale_vector.append(objective_gradient_vector.norm_inf()/constraint_gradient_vector[current_iteration].norm_inf())
                KM.Logger.Print("")
                KM.Logger.PrintInfo("ShapeOpt", "Scale  = ",  scale_vector[current_iteration])
            else:
                scale_vector.append(1.0)
        constraint_value[current_iteration]=(scale_vector[current_iteration])*constraint_value[current_iteration]
        return constraint_value[current_iteration]
# ==============================================================================
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

        if self.objectives[0]["project_gradient_on_surface_normals"].GetBool():
            self.model_part_controller.ComputeUnitSurfaceNormals()
            self.model_part_controller.ProjectNodalVariableOnUnitSurfaceNormals(KSO.DF1DX)  

        #Damping Variables
        self.model_part_controller.DampNodalVariableIfSpecified(KSO.DF1DX)
        for constraint in self.constraints:
            con_id=constraint["identifier"].GetString()
            gradient_variable = self.constraint_gradient_variables[con_id]["gradient"]
            if constraint["project_gradient_on_surface_normals"].GetBool():
                self.model_part_controller.ProjectNodalVariableOnUnitSurfaceNormals(gradient_variable)
            self.model_part_controller.DampNodalVariableIfSpecified(gradient_variable)
# ==============================================================================
    def __AreInitialConstraintFeasible(self,g_values,h_values):
        c_k=list()
        c_knext=list()
        c_k_eq=list()######Page 455
        c_knext_eq=list()
        if self.number_ineq >0:
            for itr in range(self.number_ineq):
                c_k.append(max(g_values[itr],0))
            
                              
        if self.number_eq >0:
            for itr in range(self.number_eq):
                c_k_eq.append(max(h_values[itr],0))
        return c_k,c_knext,c_k_eq,c_knext_eq  
# ==============================================================================
    def __Mapping(self):
        # Compute update in regular units
        self.mapper.Update()
        self.mapper.InverseMap(KSO.DF1DX, KSO.DF1DX_MAPPED) 
        #Here all the constraints are mapped
        for constraint in self.constraints:
            con_id = constraint["identifier"].GetString()
            gradient_contraint = self.constraint_gradient_variables[con_id]["gradient"]
            mapped_gradient_variable = self.constraint_gradient_variables[con_id]["mapped_gradient"]
            self.mapper.InverseMap(gradient_contraint, mapped_gradient_variable)
# ==============================================================================
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
# ==============================================================================   
    def __IdentifyNumberInequalities(self):
        for constraint in self.constraints:
            identifier = constraint["identifier"].GetString()
            if self.__InequalityConstraint(constraint)==True:
               self.number_ineq+=1
            else:
                self.number_eq+=1
# ==============================================================================
    def __InequalityConstraint(self, constraint):
        
        if constraint["type"].GetString() != "=":
            return True
        else:
            return False
# ==============================================================================
    def __AdjustStepSize(self,previous_A):
        current_a = self.step_size

        # Compare actual and estimated improvement using linear information from the previos step
        dfda1 = 0.0
        for node in self.design_surface.Nodes:
            # The following variables are not yet updated and therefore contain the information from the previos step
            s1 = node.GetSolutionStepValue(KSO.SEARCH_DIRECTION)
            dfds1 = node.GetSolutionStepValue(KSO.DADX_MAPPED)
            dfda1 += s1[0]*dfds1[0] + s1[1]*dfds1[1] + s1[2]*dfds1[2]

        f2 = self.A
        f1 = previous_A

        df_actual = self.A - previous_A
        df_estimated = current_a*dfda1

        # Adjust step size if necessary
        if f2 < f1:
            estimation_error = (df_actual-df_estimated)/df_actual

            # Increase step size if estimation based on linear extrapolation matches the actual improvement within a specified tolerance
            if abs(estimation_error) < self.estimation_tolerance:
                new_a = min(current_a*self.increase_factor, self.max_step_size)

            # Leave step size unchanged if a nonliner change in the objective is observed but still a descent direction is obtained
            else:
                new_a = current_a
        else:
            # Search approximation of optimal step using interpolation
            a = current_a
            corrected_step_size = - 0.5 * dfda1 * a**2 / (f2 - f1 - dfda1 * a )

            # Starting from the new design, and assuming an opposite gradient direction, the step size to the approximated optimum behaves reciprocal
            new_a = current_a-corrected_step_size

        self.step_size = new_a
    
# ==============================================================================
    def __LogCurrentOptimizationStep(self,outer_iteration,inner_iteration,
                                                current_lambda_g,current_lambda_h,current_p_vect_ineq,current_p_vect_eq
                                                ,total_iteration):
       additional_values_to_log = {}
       additional_values_to_log["step_size"] = self.step_size
       additional_values_to_log["outer_iteration"] = outer_iteration
       additional_values_to_log["inner_iteration"] = inner_iteration
       additional_values_to_log["augmented_value"] = self.A
       additional_values_to_log["augmented_value_relative_change"] = self.dA_relative
       additional_values_to_log["current_lambda_inequalities"] = current_lambda_g
       additional_values_to_log["current_lambda_equalities"] = current_lambda_h
       additional_values_to_log["current_penalty_factor_inequalities"] = current_p_vect_ineq
       additional_values_to_log["current_penalty_factor_equalities"] = current_p_vect_eq
       self.data_logger.LogCurrentValues(total_iteration, additional_values_to_log)
       self.data_logger.LogCurrentDesign(total_iteration)

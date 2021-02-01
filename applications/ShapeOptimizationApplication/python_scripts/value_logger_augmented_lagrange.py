# ==============================================================================
#  KratosShapeOptimizationApplication
#
#  License:         BSD License
#                   license: ShapeOptimizationApplication/license.txt
#
#  Main authors:    Geiser Armin, https://github.com/armingeiser
#
# ==============================================================================

# importing the Kratos Library
import KratosMultiphysics as KM

# Import logger base classes
from .value_logger_base import ValueLogger

# Import additional libraries
import csv
from .custom_timer import Timer

# ==============================================================================
class ValueLoggerAugmentedLagrange( ValueLogger ):
    # --------------------------------------------------------------------------
    
    def InitializeLogging( self ):
        self.number_eq=0
        self.number_ineq=0
        self.__IdentifyNumberInequalities()
        
        with open(self.complete_log_file_name, 'w') as csvfile:
            historyWriter = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            row = []
            row.append("{:>4s}".format("itr"))
            row.append("{:>13s}".format("f"))
            row.append("{:>13s}".format("df_abs[%]"))
            row.append("{:>13s}".format("df_rel[%]"))

            for itr in range(self.constraints.size()):#for itr in range(self.constraints.size()):
                con_type = self.constraints[itr]["type"].GetString()
                row.append("{:>13s}".format("c"+str(itr+1)+": "+con_type))
                row.append("{:>13s}".format("c"+str(itr+1)+"_ref"))

            
            row.append("{:>13s}".format("inner_iteration"))
            row.append("{:>13s}".format("outer_iteration"))
           
            row.append("{:>13s}".format("A_"))
            row.append("{:>13s}".format("dA_rel[%]"))
            
            for itr in range(self.number_eq):
                row.append("{:>13s}".format("l_multiplier_equality"+str(itr+1)))
                row.append("{:>13s}".format("p_factor_equality"+str(itr+1)))

            for itr in range(self.number_ineq):
                row.append("{:>13s}".format("l_multiplier_inequality"+str(itr+1)))
                row.append("{:>13s}".format("p_factor_inequality"+str(itr+1)))


            row.append("{:>13s}".format("step_size"))
            row.append("{:>25s}".format("time_stamp"))
            
            historyWriter.writerow(row)

    # --------------------------------------------------------------------------
    def _WriteCurrentValuesToConsole( self ):
        objective_id = self.objectives[0]["identifier"].GetString()
        KM.Logger.Print("")
        KM.Logger.PrintInfo("ShapeOpt", "Current value of objective = ", "{:> .5E}".format(self.history["response_value"][objective_id][self.current_index]))
        
        KM.Logger.PrintInfo("ShapeOpt", "Absolute change of objective = ","{:> .5E}".format(self.history["abs_change_objective"][self.current_index])," [%]")
        KM.Logger.PrintInfo("ShapeOpt", "Relative change of objective = ","{:> .5E}".format(self.history["rel_change_objective"][self.current_index])," [%]\n")

        for itr in range(self.constraints.size()):
            constraint_id = self.constraints[itr]["identifier"].GetString()
            KM.Logger.PrintInfo("ShapeOpt", "Value of C"+str(itr+1)+" = ", "{:> .5E}".format(self.history["response_value"][constraint_id][self.current_index]))
        
        KM.Logger.PrintInfo("\nShapeOpt", "Current value of Augmented Function = ", "{:> .5E}".format(self.history["augmented_value"][self.current_index]))
        KM.Logger.PrintInfo("ShapeOpt", "Relative change of Augmented Function = ","{:> .5E}".format(self.history["augmented_value_relative_change"][self.current_index])," [%]\n")


        for itr in range(len(self.history["current_penalty_factor_equalities"][self.current_index])):
            KM.Logger.PrintInfo("ShapeOpt", "Current equality Lambda "+str(itr+1)+" = ", "{:> .5E}".format(self.history["current_lambda_equalities"][self.current_index][itr]))
            KM.Logger.PrintInfo("ShapeOpt", "Current equality penalty factor "+str(itr+1)+" = ", "{:> .5E}".format(self.history["current_penalty_factor_equalities"][self.current_index][itr]),"\n")
            
        for itr in range(len(self.history["current_penalty_factor_inequalities"][self.current_index])):
            KM.Logger.PrintInfo("ShapeOpt", "Current inequality Lambda "+str(itr+1)+" = ", "{:> .5E}".format(self.history["current_lambda_inequalities"][self.current_index][itr]))
            KM.Logger.PrintInfo("ShapeOpt", "Current inequality penalty factor "+str(itr+1)+" = ", "{:> .5E}".format(self.history["current_penalty_factor_inequalities"][self.current_index][itr]))
    # --------------------------------------------------------------------------
    def _WriteCurrentValuesToFile( self ):
        with open(self.complete_log_file_name, 'a') as csvfile:
            historyWriter = csv.writer(csvfile, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
            row = []
            row.append("{:>4d}".format(self.current_index))

            objective_id = self.objectives[0]["identifier"].GetString()
            row.append(" {:> .5E}".format(self.history["response_value"][objective_id][self.current_index]))
            row.append(" {:> .5E}".format(self.history["abs_change_objective"][self.current_index]))
            row.append(" {:> .5E}".format(self.history["rel_change_objective"][self.current_index]))

            for itr in range(self.constraints.size()):
                constraint_id = self.constraints[itr]["identifier"].GetString()
                row.append(" {:> .5E}".format(self.history["response_value"][constraint_id][self.current_index]))
                row.append(" {:> .5E}".format(self.communicator.getReferenceValue(constraint_id)))

            row.append("{:> .5E}".format(self.history["inner_iteration"][self.current_index]))
            row.append("{:> .5E}".format(self.history["outer_iteration"][self.current_index]))
           
            row.append("{:> .5E}".format(self.history["augmented_value"][self.current_index]))
            row.append("{:> .5E}".format(self.history["augmented_value_relative_change"][self.current_index]))
            

            for itr in range(len(self.history["current_penalty_factor_equalities"][self.current_index])):
                row.append("{:> .5E}".format(self.history["current_lambda_equalities"][self.current_index][itr]))
                row.append("{:> .5E}".format(self.history["current_penalty_factor_equalities"][self.current_index][itr]))
            
            for itr in range(len(self.history["current_penalty_factor_inequalities"][self.current_index])):
                row.append("{:> .5E}".format(self.history["current_lambda_inequalities"][self.current_index][itr]))
                row.append("{:> .5E}".format(self.history["current_penalty_factor_inequalities"][self.current_index][itr]))
               




            row.append(" {:> .7E}".format(self.history["step_size"][self.current_index]))
            row.append("{:>25}".format(Timer().GetTimeStamp()))
            historyWriter.writerow(row)

# ==============================================================================
    def __IdentifyNumberInequalities(self):
        for constraint in self.constraints:
            identifier = constraint["identifier"].GetString()
            if self.__InequalityConstraint(constraint)==True:
                self.number_ineq+=1
            else:
                self.number_eq+=1

    def __InequalityConstraint(self, constraint):
        if constraint["type"].GetString() != "=":
            return True
        else:
            return False
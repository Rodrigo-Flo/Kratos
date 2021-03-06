import KratosMultiphysics
from KratosMultiphysics import Logger
Logger.GetDefaultOutput().SetSeverity(Logger.Severity.INFO)
import KratosMultiphysics.DEMApplication

import KratosMultiphysics.DEMApplication.main_script as Main

model = KratosMultiphysics.Model()
solution = Main.Solution(model)
solution.Run()

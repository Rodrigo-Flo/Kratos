//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics 
//
//  License:		 BSD License 
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Riccardo Rossi
//                    
//

#if !defined(KRATOS_FIND_NODAL_H_PROCESS_INCLUDED )
#define  KRATOS_FIND_NODAL_H_PROCESS_INCLUDED

// System includes
#include <limits>

// External includes

// Project includes
#include "includes/define.h"
#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{

///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{


///@}
///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{

/// Short class definition.
/** Detail class definition.
	Calculate the NODAL_H for all the nodes by means of the element sides minimum length
*/

class FindNodalHProcess : public Process
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of FindNodalHProcess
    KRATOS_CLASS_POINTER_DEFINITION(FindNodalHProcess);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    FindNodalHProcess(ModelPart& model_part) : mrModelPart(model_part)
    {
    }

    /// Destructor.
    ~FindNodalHProcess() override
    {
    }


    ///@}
    ///@name Operators
    ///@{

    void operator()()
    {
        Execute();
    }


    ///@}
    ///@name Operations
    ///@{

    void Execute() override
    {
        KRATOS_TRY
        
        // Check if variables are available       
        KRATOS_ERROR_IF_NOT(mrModelPart.NodesBegin()->SolutionStepsDataHas( NODAL_H )) << "Variable NODAL_H not in the model part!";
        
        #pragma omp parallel for 
        for(int i=0; i<static_cast<int>(mrModelPart.Nodes().size()); ++i) {
            auto it_node = mrModelPart.NodesBegin() + i;
            it_node->GetSolutionStepValue(NODAL_H, 0) = std::numeric_limits<double>::max();
        }
        
        for(unsigned int i=0; i<mrModelPart.Elements().size(); ++i) {
            auto it_element = mrModelPart.ElementsBegin() + i;
            auto& geom = it_element->GetGeometry();
            
            for(unsigned int k=0; k<geom.size()-1; ++k) {
                double& h1 = geom[k].FastGetSolutionStepValue(NODAL_H);
                for(unsigned int l=k+1; l<geom.size(); ++l) {
                    double hedge = norm_2(geom[l].Coordinates() - geom[k].Coordinates());
                    double& h2 = geom[l].FastGetSolutionStepValue(NODAL_H);
                    
                    // Get minimum between the existent value and the considered edge length 
                    geom[k].FastGetSolutionStepValue(NODAL_H) = std::min(h1, hedge);
                    geom[l].FastGetSolutionStepValue(NODAL_H) = std::min(h2, hedge);
                }
            }
        }
        
        mrModelPart.GetCommunicator().SynchronizeCurrentDataToMin(NODAL_H);

        KRATOS_CATCH("")
    }

    ///@}
    ///@name Access
    ///@{


    ///@}
    ///@name Inquiry
    ///@{


    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const override
    {
        return "FindNodalHProcess";
    }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << "FindNodalHProcess";
    }

    /// Print object's data.
    void PrintData(std::ostream& rOStream) const override
    {
    }


    ///@}
    ///@name Friends
    ///@{


    ///@}

protected:
    ///@name Protected static Member Variables
    ///@{


    ///@}
    ///@name Protected member Variables
    ///@{


    ///@}
    ///@name Protected Operators
    ///@{


    ///@}
    ///@name Protected Operations
    ///@{


    ///@}
    ///@name Protected  Access
    ///@{


    ///@}
    ///@name Protected Inquiry
    ///@{


    ///@}
    ///@name Protected LifeCycle
    ///@{


    ///@}

private:
    ///@name Static Member Variables
    ///@{


    ///@}
    ///@name Member Variables
    ///@{
    
    ModelPart& mrModelPart;
    double mMinH;

    ///@}
    ///@name Private Operators
    ///@{

    ///@}
    ///@name Private Operations
    ///@{


    ///@}
    ///@name Private  Access
    ///@{


    ///@}
    ///@name Private Inquiry
    ///@{


    ///@}
    ///@name Un accessible methods
    ///@{

    /// Assignment operator.
    FindNodalHProcess& operator=(FindNodalHProcess const& rOther);

    /// Copy constructor.
    //FindNodalHProcess(FindNodalHProcess const& rOther);


    ///@}

}; // Class FindNodalHProcess

///@}

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{


/// input stream function
inline std::istream& operator >> (std::istream& rIStream,
                                  FindNodalHProcess& rThis);

/// output stream function
inline std::ostream& operator << (std::ostream& rOStream,
                                  const FindNodalHProcess& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}
///@}


}  // namespace Kratos.

#endif // KRATOS_FIND_NODAL_H_PROCESS_INCLUDED  defined 



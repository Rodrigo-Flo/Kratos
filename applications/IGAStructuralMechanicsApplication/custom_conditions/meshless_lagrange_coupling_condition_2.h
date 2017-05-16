#if !defined(KRATOS_MESHLESS_LAGRANGE_COUPLING_CONDITION_2_H_INCLUDED )
#define  KRATOS_MESHLESS_LAGRANGE_COUPLING_CONDITION_2_H_INCLUDED



// System includes


// External includes
#include "boost/smart_ptr.hpp"


// Project includes
#include "includes/define.h"
#include "includes/condition.h"
#include "includes/ublas_interface.h"
#include "includes/variables.h"

#include "custom_conditions/meshless_base_condition.h"
#include "custom_conditions/meshless_base_coupling_condition.h"

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
*/
class MeshlessLagrangeCouplingCondition2
    : public MeshlessBaseCouplingCondition
{
public:
    ///@name Type Definitions
    ///@{

    /// Counted pointer of MeshlessLagrangeCouplingCondition2
    KRATOS_CLASS_POINTER_DEFINITION(MeshlessLagrangeCouplingCondition2);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    MeshlessLagrangeCouplingCondition2(IndexType NewId, GeometryType::Pointer pGeometry);
    MeshlessLagrangeCouplingCondition2(IndexType NewId, GeometryType::Pointer pGeometry,  PropertiesType::Pointer pProperties);

    /// Destructor.
    virtual ~MeshlessLagrangeCouplingCondition2();


    ///@}
    ///@name Operators
    ///@{


    ///@}
    ///@name Operations
    ///@{

    Condition::Pointer Create(IndexType NewId, NodesArrayType const& ThisNodes,  PropertiesType::Pointer pProperties) const override;
	
	//void Initialize();

    void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector, ProcessInfo& rCurrentProcessInfo) override;

    void CalculateRightHandSide(VectorType& rRightHandSideVector, ProcessInfo& rCurrentProcessInfo) override;

    void EquationIdVector(EquationIdVectorType& rResult, ProcessInfo& rCurrentProcessInfo) override;

    void GetDofList(DofsVectorType& ElementalDofList,ProcessInfo& CurrentProcessInfo) override;

    ///@}
    ///@name Access
    ///@{


    ///@}
    ///@name Inquiry
    ///@{


    ///@}
    ///@name Input and output
    ///@{

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


    ///@}
    ///@name Serialization
    ///@{
    friend class Serializer;

    // A private default constructor necessary for serialization
    MeshlessLagrangeCouplingCondition2() : MeshlessBaseCouplingCondition()
    {
    }


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
    //MeshlessLagrangeCouplingCondition2& operator=(const MeshlessLagrangeCouplingCondition2& rOther);

    /// Copy constructor.
    //MeshlessLagrangeCouplingCondition2(const MeshlessLagrangeCouplingCondition2& rOther);


    ///@}

}; // Class MeshlessLagrangeCouplingCondition2

///@}

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{


/// input stream function
/*  inline std::istream& operator >> (std::istream& rIStream,
				    MeshlessLagrangeCouplingCondition2& rThis);
*/
/// output stream function
/*  inline std::ostream& operator << (std::ostream& rOStream,
				    const MeshlessLagrangeCouplingCondition2& rThis)
    {
      rThis.PrintInfo(rOStream);
      rOStream << std::endl;
      rThis.PrintData(rOStream);

      return rOStream;
    }*/
///@}

}  // namespace Kratos.

#endif // KRATOS_MESHLESS_LAGRANGE_COUPLING_CONDITION_2_H_INCLUDED  defined 



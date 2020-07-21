//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:   Jordi Cotela
//                  Suneth Warnakulasuriya
//

#if !defined(KRATOS_GENERIC_RESIDUAL_BASED_BOSSAK_SCALAR_TRANSPORT_SCHEME_H_INCLUDED)
#define KRATOS_GENERIC_RESIDUAL_BASED_BOSSAK_SCALAR_TRANSPORT_SCHEME_H_INCLUDED

// System includes
#include <sstream>
#include <vector>

// External includes

// Project includes
#include "custom_strategies/relaxed_dof_updater.h"
#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategies/schemes/scheme.h"
#include "utilities/time_discretization.h"

// Application includes
#include "custom_strategies/generic_residual_based_steady_scalar_transport_scheme.h"

namespace Kratos
{
///@name Kratos Classes
///@{

/// A scheme for steady and dynamic equations, using Bossak time integration.
/**
 * It can be used for either first- or second-order time derivatives. Elements
 * and conditions must provide a specialization of SchemeExtension via
 * their data value container, which allows the scheme to operate independently
 * of the variable arrangements in the element or condition.
 */
template <class TSparseSpace, class TDenseSpace>
class GenericResidualBasedBossakScalarTransportScheme
    : public GenericResidualBasedSteadyScalarTransportScheme<TSparseSpace, TDenseSpace>
{
public:
    ///@name Type Definitions
    ///@{

    KRATOS_CLASS_POINTER_DEFINITION(GenericResidualBasedBossakScalarTransportScheme);

    using BaseType =
        GenericResidualBasedSteadyScalarTransportScheme<TSparseSpace, TDenseSpace>;

    using SystemMatrixType = typename BaseType::TSystemMatrixType;

    using SystemVectorType = typename BaseType::TSystemVectorType;

    using LocalSystemVectorType = typename BaseType::LocalSystemVectorType;

    using LocalSystemMatrixType = typename BaseType::LocalSystemMatrixType;

    using DofsArrayType = typename BaseType::DofsArrayType;

    using NodeType = ModelPart::NodeType;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Constructor.

    GenericResidualBasedBossakScalarTransportScheme(const double AlphaBossak,
                                                    const double RelaxationFactor,
                                                    const Variable<double>& rScalarVariable,
                                                    const Variable<double>& rScalarRateVariable,
                                                    const Variable<double>& rRelaxedScalarRateVariable)
        : BaseType(RelaxationFactor),
          mAlphaBossak(AlphaBossak),
          mrScalarVariable(rScalarVariable),
          mrScalarRateVariable(rScalarRateVariable),
          mrRelaxedScalarRateVariable(rRelaxedScalarRateVariable)
    {
        // Allocate auxiliary memory.
        const int num_threads = OpenMPUtils::GetNumThreads();

        mMassMatrix.resize(num_threads);
        mSecondDerivativeValuesVector.resize(num_threads);
        mSecondDerivativeValuesVectorOld.resize(num_threads);
    }

    /// Destructor.
    ~GenericResidualBasedBossakScalarTransportScheme() override = default;

    ///@}
    ///@name Operations
    ///@{

    void InitializeSolutionStep(ModelPart& rModelPart,
                                SystemMatrixType& rA,
                                SystemVectorType& rDx,
                                SystemVectorType& rb) override
    {
        KRATOS_TRY;

        BaseType::InitializeSolutionStep(rModelPart, rA, rDx, rb);

        const double delta_time = rModelPart.GetProcessInfo()[DELTA_TIME];

        KRATOS_ERROR_IF(delta_time < std::numeric_limits<double>::epsilon())
            << "detected delta_time = 0 in the Bossak Scheme ... "
               "check if the time step is created correctly for "
               "the current model part.";

        GenericResidualBasedBossakScalarTransportScheme::CalculateBossakConstants(
            mBossak, mAlphaBossak, delta_time);

#pragma omp critical
        {
            rModelPart.GetProcessInfo()[BOSSAK_ALPHA] = mBossak.Alpha;
        }

        KRATOS_CATCH("");
    }

    void Update(ModelPart& rModelPart,
                DofsArrayType& rDofSet,
                SystemMatrixType& rA,
                SystemVectorType& rDx,
                SystemVectorType& rb) override
    {
        KRATOS_TRY;

        BaseType::Update(rModelPart, rDofSet, rA, rDx, rb);

        this->UpdateScalarRateVariables(rModelPart);

        KRATOS_CATCH("");
    }

    int Check(ModelPart& rModelPart) override
    {
        KRATOS_TRY

        int value = BaseType::Check(rModelPart);

        KRATOS_ERROR_IF(!rModelPart.HasNodalSolutionStepVariable(mrScalarVariable))
            << mrScalarVariable.Name() << " not in nodal solution step variable list of "
            << rModelPart.Name() << ".\n";
        KRATOS_ERROR_IF(!rModelPart.HasNodalSolutionStepVariable(mrScalarRateVariable))
            << mrScalarRateVariable.Name() << " not in nodal solution step variable list of "
            << rModelPart.Name() << ".\n";
        KRATOS_ERROR_IF(!rModelPart.HasNodalSolutionStepVariable(mrRelaxedScalarRateVariable))
            << mrRelaxedScalarRateVariable.Name() << " not in nodal solution step variable list of "
            << rModelPart.Name() << ".\n";

        return value;
        KRATOS_CATCH("");
    }

    void CalculateSystemContributions(Element& rElement,
                                      LocalSystemMatrixType& rLHS_Contribution,
                                      LocalSystemVectorType& rRHS_Contribution,
                                      Element::EquationIdVectorType& rEquationIdVector,
                                      const ProcessInfo& rCurrentProcessInfo) override
    {
        CalculateDynamicSystem<Element>(rElement, rLHS_Contribution, rRHS_Contribution,
                                        rEquationIdVector, rCurrentProcessInfo);
    }

    void CalculateRHSContribution(Element& rElement,
                                  LocalSystemVectorType& rRHS_Contribution,
                                  Element::EquationIdVectorType& rEquationIdVector,
                                  const ProcessInfo& rCurrentProcessInfo) override
    {
        CalculateDynamicRHS<Element>(rElement, rRHS_Contribution,
                                     rEquationIdVector, rCurrentProcessInfo);
    }

    void CalculateSystemContributions(Condition& rCondition,
                                      LocalSystemMatrixType& rLHS_Contribution,
                                      LocalSystemVectorType& rRHS_Contribution,
                                      Condition::EquationIdVectorType& rEquationIdVector,
                                      const ProcessInfo& rCurrentProcessInfo) override
    {
        CalculateDynamicSystem<Condition>(rCondition, rLHS_Contribution, rRHS_Contribution,
                                          rEquationIdVector, rCurrentProcessInfo);
    }

    void CalculateRHSContribution(Condition& rCondition,
                                  LocalSystemVectorType& rRHS_Contribution,
                                  Element::EquationIdVectorType& rEquationIdVector,
                                  const ProcessInfo& rCurrentProcessInfo) override
    {
        CalculateDynamicRHS<Condition>(rCondition, rRHS_Contribution,
                                       rEquationIdVector, rCurrentProcessInfo);
    }

    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const override
    {
        std::stringstream msg;
        msg << "Using generic residual based bossak scalar transport scheme "
               "with\n"
            << "     Scalar variable             : " << mrScalarVariable.Name() << "\n"
            << "     Scalar rate variable        : " << mrScalarRateVariable.Name() << "\n"
            << "     Relaxed scalar rate variable: "
            << mrRelaxedScalarRateVariable.Name() << "\n"
            << "     Relaxation factor           : " << this->mRelaxationFactor;

        return msg.str();
    }

    ///@}

private:
    ///@name Member Variables
    ///@{

    struct BossakConstants
    {
        double Alpha;
        double Gamma;
        double C0;
        double C2;
        double C3;
    };

    std::vector<LocalSystemVectorType> mSecondDerivativeValuesVectorOld;
    std::vector<LocalSystemVectorType> mSecondDerivativeValuesVector;
    std::vector<LocalSystemMatrixType> mMassMatrix;
    std::vector<LocalSystemMatrixType> mDampingMatrix;

    const double mAlphaBossak;

    const Variable<double>& mrScalarVariable;
    const Variable<double>& mrScalarRateVariable;
    const Variable<double>& mrRelaxedScalarRateVariable;

    BossakConstants mBossak;

    ///@}
    ///@name Private Operations
    ///@{

    static void CalculateBossakConstants(BossakConstants& rBossakConstants,
                                         const double Alpha,
                                         const double DeltaTime)
    {
        TimeDiscretization::Bossak bossak(Alpha, 0.25, 0.5);
        rBossakConstants.Alpha = bossak.GetAlphaM();
        rBossakConstants.Gamma = bossak.GetGamma();

        rBossakConstants.C0 =
            (1.0 - rBossakConstants.Alpha) / (rBossakConstants.Gamma * DeltaTime);
        rBossakConstants.C2 = 1.0 / (rBossakConstants.Gamma * DeltaTime);
        rBossakConstants.C3 = (1.0 - rBossakConstants.Gamma) / rBossakConstants.Gamma;
    }

    template <class TItemType>
    void AddMassMatrixToRHS(TItemType& rItem, LocalSystemVectorType& rRHS_Contribution, const int ThreadId)
    {
        rItem.GetSecondDerivativesVector(mSecondDerivativeValuesVector[ThreadId], 0);
        (mSecondDerivativeValuesVector[ThreadId]) *= (1.00 - mBossak.Alpha);
        rItem.GetSecondDerivativesVector(mSecondDerivativeValuesVectorOld[ThreadId], 1);
        noalias(mSecondDerivativeValuesVector[ThreadId]) +=
            mBossak.Alpha * mSecondDerivativeValuesVectorOld[ThreadId];

        noalias(rRHS_Contribution) -=
            prod(mMassMatrix[ThreadId], mSecondDerivativeValuesVector[ThreadId]);
    }

    template <class TItemType>
    void CalculateDynamicSystem(TItemType& rItem,
                                LocalSystemMatrixType& rLHS_Contribution,
                                LocalSystemVectorType& rRHS_Contribution,
                                typename TItemType::EquationIdVectorType& rEquationIdVector,
                                const ProcessInfo& rCurrentProcessInfo)
    {
        KRATOS_TRY;

        BaseType::CalculateSystemContributions(rItem, rLHS_Contribution, rRHS_Contribution,
                                               rEquationIdVector, rCurrentProcessInfo);

        const int k = OpenMPUtils::ThisThread();

        rItem.CalculateMassMatrix(mMassMatrix[k], rCurrentProcessInfo);
        // adding mass contribution to the dynamic stiffness
        if (mMassMatrix[k].size1() != 0) // if M matrix declared
        {
            AddMassMatrixToRHS<TItemType>(rItem, rRHS_Contribution, k);

            noalias(rLHS_Contribution) += mBossak.C0 * mMassMatrix[k];
        }

        KRATOS_CATCH("");
    }

    template <class TItemType>
    void CalculateDynamicRHS(TItemType& rItem,
                             LocalSystemVectorType& rRHS_Contribution,
                             typename TItemType::EquationIdVectorType& rEquationIdVector,
                             const ProcessInfo& rCurrentProcessInfo)
    {
        KRATOS_TRY;

        const int k = OpenMPUtils::ThisThread();

        this->CalculateDampingSystem(rItem, this->mDampingMatrix[k], rRHS_Contribution,
                                     rEquationIdVector, rCurrentProcessInfo, k);

        rItem.CalculateMassMatrix(mMassMatrix[k], rCurrentProcessInfo);
        // adding mass contribution to the dynamic stiffness
        if (mMassMatrix[k].size1() != 0) // if M matrix declared
        {
            AddMassMatrixToRHS(rItem, rRHS_Contribution, k);
        }

        KRATOS_CATCH("");
    }

    void UpdateScalarRateVariables(ModelPart& rModelPart)
    {
        const int number_of_nodes = rModelPart.NumberOfNodes();

#pragma omp parallel for
        for (int i_node = 0; i_node < number_of_nodes; ++i_node)
        {
            NodeType& r_node = *(rModelPart.NodesBegin() + i_node);
            double& r_current_rate = r_node.FastGetSolutionStepValue(mrScalarRateVariable);
            const double old_rate =
                r_node.FastGetSolutionStepValue(mrScalarRateVariable, 1);
            const double current_value = r_node.FastGetSolutionStepValue(mrScalarVariable);
            const double old_value =
                r_node.FastGetSolutionStepValue(mrScalarVariable, 1);

            // update scalar rate variable
            r_current_rate = mBossak.C2 * (current_value - old_value) - mBossak.C3 * old_rate;

            // update relaxed scalar rate variable
            r_node.FastGetSolutionStepValue(mrRelaxedScalarRateVariable) =
                this->mAlphaBossak * old_rate + (1.0 - this->mAlphaBossak) * r_current_rate;
        }
    }

    ///@}

}; /* Class GenericResidualBasedBossakScalarTransportScheme */

///@}

} /* namespace Kratos.*/

#endif /* KRATOS_GENERIC_RESIDUAL_BASED_BOSSAK_SCALAR_TRANSPORT_SCHEME_H_INCLUDED defined */
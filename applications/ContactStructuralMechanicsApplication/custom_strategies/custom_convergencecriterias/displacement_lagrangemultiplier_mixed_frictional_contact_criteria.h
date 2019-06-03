// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:             BSD License
//                                       license: StructuralMechanicsApplication/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

#if !defined(KRATOS_DISPLACEMENT_LAGRANGE_MULTIPLIER_MIXED_FRICTIONAL_CONTACT_CRITERIA_H)
#define KRATOS_DISPLACEMENT_LAGRANGE_MULTIPLIER_MIXED_FRICTIONAL_CONTACT_CRITERIA_H

/* System includes */

/* External includes */

/* Project includes */
#include "utilities/table_stream_utility.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"
#include "utilities/color_utilities.h"

namespace Kratos
{
///@addtogroup ContactStructuralMechanicsApplication
///@{

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

///@name Kratos Classes
///@{

/**
 * @class DisplacementLagrangeMultiplierMixedFrictionalContactCriteria
 * @ingroup ContactStructuralMechanicsApplication
 * @brief Convergence criteria for contact problems
 * @details This class implements a convergence control based on nodal displacement and
 * lagrange multiplier values. The error is evaluated separately for each of them, and
 * relative and absolute tolerances for both must be specified.
 * @author Vicente Mataix Ferrandiz
 */
template<   class TSparseSpace,
            class TDenseSpace >
class DisplacementLagrangeMultiplierMixedFrictionalContactCriteria
    : public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

    ///@name Type Definitions
    ///@{

    /// Pointer definition of DisplacementLagrangeMultiplierMixedFrictionalContactCriteria
    KRATOS_CLASS_POINTER_DEFINITION( DisplacementLagrangeMultiplierMixedFrictionalContactCriteria );

    /// Local Flags
    KRATOS_DEFINE_LOCAL_FLAG( ENSURE_CONTACT );
    KRATOS_DEFINE_LOCAL_FLAG( PRINTING_OUTPUT );
    KRATOS_DEFINE_LOCAL_FLAG( TABLE_IS_INITIALIZED );
    KRATOS_DEFINE_LOCAL_FLAG( INITIAL_RESIDUAL_IS_SET );

    /// The base class definition (and it subclasses)
    typedef ConvergenceCriteria< TSparseSpace, TDenseSpace > BaseType;
    typedef typename BaseType::TDataType                    TDataType;
    typedef typename BaseType::DofsArrayType            DofsArrayType;
    typedef typename BaseType::TSystemMatrixType    TSystemMatrixType;
    typedef typename BaseType::TSystemVectorType    TSystemVectorType;

    /// The sparse space used
    typedef TSparseSpace                              SparseSpaceType;

    /// The r_table stream definition TODO: Replace by logger
    typedef TableStreamUtility::Pointer       TablePrinterPointerType;

    /// The index type definition
    typedef std::size_t                                     IndexType;

    /// The key type definition
    typedef std::size_t                                       KeyType;

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Default constructor.
     * @param DispRatioTolerance Relative tolerance for displacement residual error
     * @param DispAbsTolerance Absolute tolerance for displacement residual error
     * @param LMRatioTolerance Relative tolerance for lagrange multiplier residual  error
     * @param LMAbsTolerance Absolute tolerance for lagrange multiplier residual error
     * @param EnsureContact To check if the contact is lost
     * @param pTable The pointer to the output r_table
     * @param PrintingOutput If the output is going to be printed in a txt file
     */
    explicit DisplacementLagrangeMultiplierMixedFrictionalContactCriteria(
        const TDataType DispRatioTolerance,
        const TDataType DispAbsTolerance,
        const TDataType LMNormalRatioTolerance,
        const TDataType LMNormalAbsTolerance,
        const TDataType LMTangentRatioTolerance,
        const TDataType LMTangentAbsTolerance,
        const bool EnsureContact = false,
        const bool PrintingOutput = false
        )
        : BaseType()
    {
        // Set local flags
        mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::ENSURE_CONTACT, EnsureContact);
        mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::PRINTING_OUTPUT, PrintingOutput);
        mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::TABLE_IS_INITIALIZED, false);
        mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::INITIAL_RESIDUAL_IS_SET, false);

        // The displacement residual
        mDispRatioTolerance = DispRatioTolerance;
        mDispAbsTolerance = DispAbsTolerance;

        // The normal contact residual
        mLMNormalRatioTolerance = LMNormalRatioTolerance;
        mLMNormalAbsTolerance = LMNormalAbsTolerance;

        // The tangent contact residual
        mLMTangentRatioTolerance = LMTangentRatioTolerance;
        mLMTangentAbsTolerance = LMTangentAbsTolerance;
    }

    /**
     * @brief Default constructor (parameters)
     * @param ThisParameters The configuration parameters
     */
    explicit DisplacementLagrangeMultiplierMixedFrictionalContactCriteria( Parameters ThisParameters = Parameters(R"({})"))
        : BaseType()
    {
        // The default parameters
        Parameters default_parameters = Parameters(R"(
        {
            "ensure_contact"                                     : false,
            "print_convergence_criterion"                        : false,
            "residual_relative_tolerance"                        : 1.0e-4,
            "residual_absolute_tolerance"                        : 1.0e-9,
            "contact_displacement_relative_tolerance"            : 1.0e-4,
            "contact_displacement_absolute_tolerance"            : 1.0e-9,
            "frictional_contact_displacement_relative_tolerance" : 1.0e-4,
            "frictional_contact_displacement_absolute_tolerance" : 1.0e-9
        })" );

        ThisParameters.ValidateAndAssignDefaults(default_parameters);

        // The displacement residual
        mDispRatioTolerance = ThisParameters["residual_relative_tolerance"].GetDouble();
        mDispAbsTolerance = ThisParameters["residual_absolute_tolerance"].GetDouble();

        // The normal contact solution
        mLMNormalRatioTolerance =  ThisParameters["contact_displacement_relative_tolerance"].GetDouble();
        mLMNormalAbsTolerance =  ThisParameters["contact_displacement_absolute_tolerance"].GetDouble();

        // The tangent contact solution
        mLMTangentRatioTolerance =  ThisParameters["frictional_contact_displacement_relative_tolerance"].GetDouble();
        mLMTangentAbsTolerance =  ThisParameters["frictional_contact_displacement_absolute_tolerance"].GetDouble();

        // Set local flags
        mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::ENSURE_CONTACT, ThisParameters["ensure_contact"].GetBool());
        mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::PRINTING_OUTPUT, ThisParameters["print_convergence_criterion"].GetBool());
        mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::TABLE_IS_INITIALIZED, false);
        mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::INITIAL_RESIDUAL_IS_SET, false);
    }

    //* Copy constructor.
    DisplacementLagrangeMultiplierMixedFrictionalContactCriteria( DisplacementLagrangeMultiplierMixedFrictionalContactCriteria const& rOther )
      :BaseType(rOther)
      ,mOptions(rOther.mOptions)
      ,mDispRatioTolerance(rOther.mDispRatioTolerance)
      ,mDispAbsTolerance(rOther.mDispAbsTolerance)
      ,mDispInitialResidualNorm(rOther.mDispInitialResidualNorm)
      ,mDispCurrentResidualNorm(rOther.mDispCurrentResidualNorm)
      ,mLMNormalRatioTolerance(rOther.mLMNormalRatioTolerance)
      ,mLMNormalAbsTolerance(rOther.mLMNormalAbsTolerance)
      ,mLMTangentRatioTolerance(rOther.mLMNormalRatioTolerance)
      ,mLMTangentAbsTolerance(rOther.mLMNormalAbsTolerance)
    {
    }

    /// Destructor.
    ~DisplacementLagrangeMultiplierMixedFrictionalContactCriteria() override = default;

    ///@}
    ///@name Operators
    ///@{

    /**
     * @brief Compute relative and absolute error.
     * @param rModelPart Reference to the ModelPart containing the contact problem.
     * @param rDofSet Reference to the container of the problem's degrees of freedom (stored by the BuilderAndSolver)
     * @param rA System matrix (unused)
     * @param rDx Vector of results (variations on nodal variables)
     * @param rb RHS vector (residual)
     * @return true if convergence is achieved, false otherwise
     */
    bool PostCriteria(
        ModelPart& rModelPart,
        DofsArrayType& rDofSet,
        const TSystemMatrixType& rA,
        const TSystemVectorType& rDx,
        const TSystemVectorType& rb
        ) override
    {
        if (SparseSpaceType::Size(rb) != 0) { //if we are solving for something
            // Initialize
            TDataType disp_residual_solution_norm = 0.0, normal_lm_solution_norm = 0.0, normal_lm_increase_norm = 0.0, tangent_lm_solution_norm = 0.0, tangent_lm_increase_norm = 0.0;
            IndexType disp_dof_num(0),lm_dof_num(0);

            // The nodes array
            auto& r_nodes_array = rModelPart.Nodes();

            // First iterator
            const auto it_dof_begin = rDofSet.begin();

            // Auxiliar values
            std::size_t dof_id = 0;
            TDataType residual_dof_value = 0.0, dof_value = 0.0, dof_incr = 0.0;

            // Loop over Dofs
            #pragma omp parallel for reduction(+:disp_residual_solution_norm,normal_lm_solution_norm,normal_lm_increase_norm,disp_dof_num,lm_dof_num,dof_id,residual_dof_value,dof_value,dof_incr)
            for (int i = 0; i < static_cast<int>(rDofSet.size()); i++) {
                auto it_dof = it_dof_begin + i;

                if (it_dof->IsFree()) {
                    dof_id = it_dof->EquationId();

                    const auto curr_var = it_dof->GetVariable();
                    if (curr_var == VECTOR_LAGRANGE_MULTIPLIER_X) {
                        // The normal of the node (TODO: how to solve this without accesing all the time to the database?)
                        const auto it_node = r_nodes_array.find(it_dof->Id());
                        const double normal_x = it_node->FastGetSolutionStepValue(NORMAL_X);

                        dof_value = it_dof->GetSolutionStepValue(0);
                        dof_incr = rDx[dof_id];
                        const TDataType normal_dof_value = dof_value * normal_x;
                        const TDataType normal_dof_incr = dof_incr * normal_x;

                        normal_lm_solution_norm += std::pow(normal_dof_value, 2);
                        normal_lm_increase_norm += std::pow(normal_dof_incr, 2);
                        tangent_lm_solution_norm += std::pow(dof_value - normal_dof_value, 2);
                        tangent_lm_increase_norm += std::pow(dof_incr - normal_dof_incr, 2);
                        lm_dof_num++;
                    } else if (curr_var == VECTOR_LAGRANGE_MULTIPLIER_Y) {
                        // The normal of the node (TODO: how to solve this without accesing all the time to the database?)
                        const auto it_node = r_nodes_array.find(it_dof->Id());
                        const double normal_y = it_node->FastGetSolutionStepValue(NORMAL_Y);

                        dof_value = it_dof->GetSolutionStepValue(0);
                        dof_incr = rDx[dof_id];
                        const TDataType normal_dof_value = dof_value * normal_y;
                        const TDataType normal_dof_incr = dof_incr * normal_y;

                        normal_lm_solution_norm += std::pow(normal_dof_value, 2);
                        normal_lm_increase_norm += std::pow(normal_dof_incr, 2);
                        tangent_lm_solution_norm += std::pow(dof_value - normal_dof_value, 2);
                        tangent_lm_increase_norm += std::pow(dof_incr - normal_dof_incr, 2);
                        lm_dof_num++;
                    } else if (curr_var == VECTOR_LAGRANGE_MULTIPLIER_Z) {
                        // The normal of the node (TODO: how to solve this without accesing all the time to the database?)
                        const auto it_node = r_nodes_array.find(it_dof->Id());
                        const double normal_z = it_node->FastGetSolutionStepValue(NORMAL_Z);

                        dof_value = it_dof->GetSolutionStepValue(0);
                        dof_incr = rDx[dof_id];
                        const TDataType normal_dof_value = dof_value * normal_z;
                        const TDataType normal_dof_incr = dof_incr * normal_z;

                        normal_lm_solution_norm += std::pow(normal_dof_value, 2);
                        normal_lm_increase_norm += std::pow(normal_dof_incr, 2);
                        tangent_lm_solution_norm += std::pow(dof_value - normal_dof_value, 2);
                        tangent_lm_increase_norm += std::pow(dof_incr - normal_dof_incr, 2);
                        lm_dof_num++;
                    } else {
                        residual_dof_value = rb[dof_id];
                        disp_residual_solution_norm += residual_dof_value * residual_dof_value;
                        disp_dof_num++;
                    }
                }
            }

            if(normal_lm_increase_norm == 0.0) normal_lm_increase_norm = 1.0;
            KRATOS_ERROR_IF(mOptions.Is(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::ENSURE_CONTACT) && normal_lm_solution_norm == 0.0) << "ERROR::CONTACT LOST::ARE YOU SURE YOU ARE SUPPOSED TO HAVE CONTACT?" << std::endl;

            mDispCurrentResidualNorm = disp_residual_solution_norm;
            const TDataType normal_lm_ratio = std::sqrt(normal_lm_increase_norm/normal_lm_solution_norm);
            const TDataType normal_lm_abs = std::sqrt(normal_lm_increase_norm)/ static_cast<TDataType>(lm_dof_num);
            const TDataType tangent_lm_ratio = std::sqrt(tangent_lm_increase_norm/tangent_lm_solution_norm);
            const TDataType tangent_lm_abs = std::sqrt(tangent_lm_increase_norm)/ static_cast<TDataType>(lm_dof_num);

            TDataType residual_disp_ratio;

            // We initialize the solution
            if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::INITIAL_RESIDUAL_IS_SET)) {
                mDispInitialResidualNorm = (disp_residual_solution_norm == 0.0) ? 1.0 : disp_residual_solution_norm;
                residual_disp_ratio = 1.0;
                mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::INITIAL_RESIDUAL_IS_SET, true);
            }

            // We calculate the ratio of the displacements
            residual_disp_ratio = mDispCurrentResidualNorm/mDispInitialResidualNorm;

            // We calculate the absolute norms
            TDataType residual_disp_abs = mDispCurrentResidualNorm/disp_dof_num;

            // The process info of the model part
            ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

            // We print the results // TODO: Replace for the new log
            if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
                if (r_process_info.Has(TABLE_UTILITY)) {
                    std::cout.precision(4);
                    TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
                    auto& r_table = p_table->GetTable();
                    r_table << residual_disp_ratio << mDispRatioTolerance << residual_disp_abs << mDispAbsTolerance << normal_lm_ratio << mLMNormalRatioTolerance << normal_lm_abs << mLMNormalAbsTolerance << tangent_lm_ratio << mLMTangentRatioTolerance << tangent_lm_abs << mLMTangentAbsTolerance;
                } else {
                    std::cout.precision(4);
                    if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::PRINTING_OUTPUT)) {
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << BOLDFONT("MIXED CONVERGENCE CHECK") << "\tSTEP: " << r_process_info[STEP] << "\tNL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << std::endl << std::scientific;
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << BOLDFONT("\tDISPLACEMENT: RATIO = ") << residual_disp_ratio << BOLDFONT(" EXP.RATIO = ") << mDispRatioTolerance << BOLDFONT(" ABS = ") << residual_disp_abs << BOLDFONT(" EXP.ABS = ") << mDispAbsTolerance << std::endl;
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << BOLDFONT("\tNORMAL LAGRANGE MUL: RATIO = ") << normal_lm_ratio << BOLDFONT(" EXP.RATIO = ") << mLMNormalRatioTolerance << BOLDFONT(" ABS = ") << normal_lm_abs << BOLDFONT(" EXP.ABS = ") << mLMNormalAbsTolerance << std::endl;
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << BOLDFONT("\tTANGENT LAGRANGE MUL: RATIO = ") << tangent_lm_ratio << BOLDFONT(" EXP.RATIO = ") << mLMTangentRatioTolerance << BOLDFONT(" ABS = ") << tangent_lm_abs << BOLDFONT(" EXP.ABS = ") << mLMTangentAbsTolerance << std::endl;
                    } else {
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << "MIXED CONVERGENCE CHECK" << "\tSTEP: " << r_process_info[STEP] << "\tNL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << std::endl << std::scientific;
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << "\tDISPLACEMENT: RATIO = " << residual_disp_ratio << " EXP.RATIO = " << mDispRatioTolerance << " ABS = " << residual_disp_abs << " EXP.ABS = " << mDispAbsTolerance << std::endl;
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << "\tNORMAL LAGRANGE MUL: RATIO = " << normal_lm_ratio << " EXP.RATIO = " << mLMNormalRatioTolerance << " ABS = " << normal_lm_abs << " EXP.ABS = " << mLMNormalAbsTolerance << std::endl;
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << "\tTANGENT LAGRANGE MUL: RATIO = " << tangent_lm_ratio << " EXP.RATIO = " << mLMTangentRatioTolerance << " ABS = " << tangent_lm_abs << " EXP.ABS = " << mLMTangentAbsTolerance << std::endl;
                    }
                }
            }

            // NOTE: Here we don't include the tangent counter part
            r_process_info[CONVERGENCE_RATIO] = (residual_disp_ratio > normal_lm_ratio) ? residual_disp_ratio : normal_lm_ratio;
            r_process_info[RESIDUAL_NORM] = (normal_lm_abs > mLMNormalAbsTolerance) ? normal_lm_abs : mLMNormalAbsTolerance;

            // We check if converged
            const bool disp_converged = (residual_disp_ratio <= mDispRatioTolerance || residual_disp_abs <= mDispAbsTolerance);
            const bool lm_converged = (mOptions.IsNot(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::ENSURE_CONTACT) && normal_lm_solution_norm == 0.0) ? true : (normal_lm_ratio <= mLMNormalRatioTolerance || normal_lm_abs <= mLMNormalAbsTolerance) && (tangent_lm_ratio <= mLMTangentRatioTolerance || tangent_lm_abs <= mLMTangentAbsTolerance);

            if ( disp_converged && lm_converged ) {
                if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
                    if (r_process_info.Has(TABLE_UTILITY)) {
                        TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
                        auto& r_table = p_table->GetTable();
                        if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::PRINTING_OUTPUT))
                            r_table << BOLDFONT(FGRN("       Achieved"));
                        else
                            r_table << "Achieved";
                    } else {
                        if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::PRINTING_OUTPUT))
                            KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << BOLDFONT("\tConvergence") << " is " << BOLDFONT(FGRN("achieved")) << std::endl;
                        else
                            KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << "\tConvergence is achieved" << std::endl;
                    }
                }
                return true;
            } else {
                if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
                    if (r_process_info.Has(TABLE_UTILITY)) {
                        TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
                        auto& r_table = p_table->GetTable();
                        if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::PRINTING_OUTPUT))
                            r_table << BOLDFONT(FRED("   Not achieved"));
                        else
                            r_table << "Not achieved";
                    } else {
                        if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::PRINTING_OUTPUT))
                            KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << BOLDFONT("\tConvergence") << " is " << BOLDFONT(FRED(" not achieved")) << std::endl;
                        else
                            KRATOS_INFO("DisplacementLagrangeMultiplierMixedFrictionalContactCriteria") << "\tConvergence is not achieved" << std::endl;
                    }
                }
                return false;
            }
        } else // In this case all the displacements are imposed!
            return true;
    }

    /**
     * @brief This function initialize the convergence criteria
     * @param rModelPart Reference to the ModelPart containing the contact problem. (unused)
     */
    void Initialize( ModelPart& rModelPart) override
    {
        BaseType::mConvergenceCriteriaIsInitialized = true;

        ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
        if (r_process_info.Has(TABLE_UTILITY) && mOptions.IsNot(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::TABLE_IS_INITIALIZED)) {
            TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
            auto& r_table = p_table->GetTable();
            r_table.AddColumn("DP RATIO", 10);
            r_table.AddColumn("EXP. RAT", 10);
            r_table.AddColumn("ABS", 10);
            r_table.AddColumn("EXP. ABS", 10);
            r_table.AddColumn("N.LM RATIO", 10);
            r_table.AddColumn("EXP. RAT", 10);
            r_table.AddColumn("ABS", 10);
            r_table.AddColumn("EXP. ABS", 10);
            r_table.AddColumn("T.LM RATIO", 10);
            r_table.AddColumn("EXP. RAT", 10);
            r_table.AddColumn("ABS", 10);
            r_table.AddColumn("EXP. ABS", 10);
            r_table.AddColumn("CONVERGENCE", 15);
            mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::TABLE_IS_INITIALIZED, true);
        }
    }

    /**
     * @brief This function initializes the solution step
     * @param rModelPart Reference to the ModelPart containing the contact problem.
     * @param rDofSet Reference to the container of the problem's degrees of freedom (stored by the BuilderAndSolver)
     * @param rA System matrix (unused)
     * @param rDx Vector of results (variations on nodal variables)
     * @param rb RHS vector (residual)
     */
    void InitializeSolutionStep(
        ModelPart& rModelPart,
        DofsArrayType& rDofSet,
        const TSystemMatrixType& rA,
        const TSystemVectorType& rDx,
        const TSystemVectorType& rb
        ) override
    {
        mOptions.Set(DisplacementLagrangeMultiplierMixedFrictionalContactCriteria::INITIAL_RESIDUAL_IS_SET, false);
    }

    ///@}
    ///@name Operations
    ///@{

    ///@}
    ///@name Acces
    ///@{

    ///@}
    ///@name Inquiry
    ///@{

    ///@}
    ///@name Friends
    ///@{

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

    Flags mOptions; /// Local flags

    TDataType mDispRatioTolerance;      /// The ratio threshold for the norm of the displacement residual
    TDataType mDispAbsTolerance;        /// The absolute value threshold for the norm of the displacement residual
    TDataType mDispInitialResidualNorm; /// The reference norm of the displacement residual
    TDataType mDispCurrentResidualNorm; /// The current norm of the displacement residual

    TDataType mLMNormalRatioTolerance; /// The ratio threshold for the norm of the LM (normal)
    TDataType mLMNormalAbsTolerance;   /// The absolute value threshold for the norm of the LM (normal)

    TDataType mLMTangentRatioTolerance; /// The ratio threshold for the norm of the LM (tangent)
    TDataType mLMTangentAbsTolerance;   /// The absolute value threshold for the norm of the LM (tangent)

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

    ///@}
    ///@name Serialization
    ///@{

    ///@name Private Inquiry
    ///@{
    ///@}

    ///@name Unaccessible methods
    ///@{
    ///@}
};  // Kratos DisplacementLagrangeMultiplierMixedFrictionalContactCriteria

///@name Local flags creation
///@{

/// Local Flags
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedFrictionalContactCriteria<TSparseSpace, TDenseSpace>::ENSURE_CONTACT(Kratos::Flags::Create(0));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedFrictionalContactCriteria<TSparseSpace, TDenseSpace>::NOT_ENSURE_CONTACT(Kratos::Flags::Create(0, false));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedFrictionalContactCriteria<TSparseSpace, TDenseSpace>::PRINTING_OUTPUT(Kratos::Flags::Create(1));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedFrictionalContactCriteria<TSparseSpace, TDenseSpace>::NOT_PRINTING_OUTPUT(Kratos::Flags::Create(1, false));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedFrictionalContactCriteria<TSparseSpace, TDenseSpace>::TABLE_IS_INITIALIZED(Kratos::Flags::Create(2));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedFrictionalContactCriteria<TSparseSpace, TDenseSpace>::NOT_TABLE_IS_INITIALIZED(Kratos::Flags::Create(2, false));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedFrictionalContactCriteria<TSparseSpace, TDenseSpace>::INITIAL_RESIDUAL_IS_SET(Kratos::Flags::Create(3));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedFrictionalContactCriteria<TSparseSpace, TDenseSpace>::NOT_INITIAL_RESIDUAL_IS_SET(Kratos::Flags::Create(3, false));
}

#endif	/* KRATOS_DISPLACEMENT_LAGRANGE_MULTIPLIER_MIXED_FRICTIONAL_CONTACT_CRITERIA_H */


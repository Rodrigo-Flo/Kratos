//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                     Kratos default license: kratos/IGAStructuralMechanicsApplication/license.txt
//
//  Main authors:    Ricky Aristio
//                   Pia Halbich
//                   Tobias Tescheamacher
//                   Riccardo Rossi
//


// System includes

// External includes

// Project includes

// Application includes
#include "custom_elements/iga_edge_cable_element.h"



namespace Kratos
{
    ///@name Initialize Functions
    ///@{

    void IgaEdgeCableElement::Initialize()
    {
        KRATOS_TRY

        const GeometryType& r_geometry = GetGeometry();

        const SizeType r_number_of_integration_points = r_geometry.IntegrationPointsNumber();

        // Prepare memory
        if (mReferenceBaseVector.size() != r_number_of_integration_points)
            mReferenceBaseVector.resize(r_number_of_integration_points);

        for (IndexType point_number = 0; point_number < r_number_of_integration_points; ++point_number)
        {
            const Matrix& r_DN_De   = r_geometry.ShapeFunctionLocalGradient(point_number);

            mReferenceBaseVector[point_number] = GetActualBaseVector(r_DN_De);          
        }

        KRATOS_CATCH("")
    }


    array_1d<double, 3> IgaEdgeCableElement::GetActualBaseVector(const Matrix& r_DN_De) 
    {
        const GeometryType& r_geometry = GetGeometry();
        const SizeType number_of_nodes = r_geometry.size();

        const Vector& t = GetProperties()[TANGENTS];
       
        array_1d<double, 3> actual_base_vector = ZeroVector(3);

        Matrix jacobian = ZeroMatrix(3, 2);
        for (unsigned int i = 0; i < number_of_nodes; i++)
        {
            for (unsigned int k = 0; k<3; k++)
            {
                for (unsigned int m = 0; m<2; m++)
                {
                    jacobian(k, m) += (r_geometry[i]).Coordinates()[k] * r_DN_De(i, m);
                }
            }
        }

        //basis vectors g1 and g2
        array_1d<double, 3> g1;
        array_1d<double, 3> g2;

        g1[0] = jacobian(0, 0);
        g2[0] = jacobian(0, 1);
        g1[1] = jacobian(1, 0);
        g2[1] = jacobian(1, 1);
        g1[2] = jacobian(2, 0);
        g2[2] = jacobian(2, 1);    

        actual_base_vector = g1 * t[0] + g2 * t[1];

        return actual_base_vector;
    }

    ///@}
    ///@name Assembly
    ///@{

    void IgaEdgeCableElement::CalculateAll(
        MatrixType& rLeftHandSideMatrix,
        VectorType& rRightHandSideVector,
        ProcessInfo& rCurrentProcessInfo,
        const bool CalculateStiffnessMatrixFlag,
        const bool CalculateResidualVectorFlag
    )
    {
        KRATOS_TRY

        const auto& r_geometry = GetGeometry();

        // definition of problem size
        const SizeType number_of_nodes = r_geometry.size();
        const SizeType mat_size = number_of_nodes * 3;

        const auto& r_integration_points = r_geometry.IntegrationPoints();

        //get properties
        const Vector& t = GetProperties()[TANGENTS];
        const double E = GetProperties()[YOUNG_MODULUS];
        const double A = GetProperties()[CROSS_AREA];
        const double prestress = GetProperties()[PRESTRESS_CAUCHY];

        for (IndexType point_number = 0; point_number < r_integration_points.size(); ++point_number) 
        {   
            // get integration data
            const double& integration_weight = r_integration_points[point_number].Weight();
            const Matrix& r_DN_De   = r_geometry.ShapeFunctionLocalGradient(point_number);

            // compute base vectors
            const array_1d<double, 3> actual_base_vector = GetActualBaseVector(r_DN_De);
    
            const double reference_a = norm_2(mReferenceBaseVector[point_number]);
            const double actual_a = norm_2(actual_base_vector);

            const double actual_aa = actual_a * actual_a;
            const double reference_aa = reference_a * reference_a;        
        
            // green-lagrange strain
            const double e11_membrane = 0.5 * (inner_prod(actual_base_vector, actual_base_vector) - inner_prod(mReferenceBaseVector[point_number], mReferenceBaseVector[point_number]));

            // normal forcereference_aa
            const double s11_membrane = prestress * A + e11_membrane * A * E / inner_prod(mReferenceBaseVector[point_number],mReferenceBaseVector[point_number]);

            for (IndexType r = 0; r < mat_size; r++)
            {
                // local node number kr and dof direction dirr
                IndexType kr = r / 3;
                IndexType dirr = r % 3;

                const double epsilon_var_r = actual_base_vector[dirr] *
                    (r_DN_De(kr, 0) * t[0] 
                    + r_DN_De(kr, 1) * t[1]) / inner_prod(mReferenceBaseVector[point_number],mReferenceBaseVector[point_number]);

                if (CalculateStiffnessMatrixFlag) {
                    for (IndexType s = 0; s < mat_size; s++)
                    {
                        // local node number ks and dof direction dirs
                        IndexType ks = s / 3;
                        IndexType dirs = s % 3;

                        const double epsilon_var_s =
                            actual_base_vector[dirs] *
                            (r_DN_De(ks, 0) * t[0]
                            + r_DN_De(ks, 1) * t[1])
                            / inner_prod(mReferenceBaseVector[point_number],mReferenceBaseVector[point_number]);

                        rLeftHandSideMatrix(r, s) = E * A * epsilon_var_r * epsilon_var_s;

                        if (dirr == dirs) {
                            const double epsilon_var_rs =
                            (r_DN_De(kr, 0) * t[0] + r_DN_De(kr, 1) * t[1]) *
                            (r_DN_De(ks, 0) * t[0] + r_DN_De(ks, 1) * t[1]) /inner_prod(mReferenceBaseVector[point_number],mReferenceBaseVector[point_number]);
                     
                            rLeftHandSideMatrix(r, s) += s11_membrane * epsilon_var_rs; 
                        }
                    }
                }
                if (CalculateResidualVectorFlag) {
                    rRightHandSideVector[r] = -s11_membrane * epsilon_var_r;
                }
            }
            if (CalculateStiffnessMatrixFlag) {
                rLeftHandSideMatrix *= reference_a * integration_weight;
            }
            if (CalculateResidualVectorFlag) {
                rRightHandSideVector *= reference_a * integration_weight;
            }
        }
        KRATOS_CATCH("");
    }

    void IgaEdgeCableElement::GetValueOnIntegrationPoints(const Variable<double>& rVariable,
        std::vector<double>& rValues, const ProcessInfo& rCurrentProcessInfo)
    {
        KRATOS_TRY;
        CalculateOnIntegrationPoints(rVariable, rValues, rCurrentProcessInfo);
        KRATOS_CATCH("")
    }

    void IgaEdgeCableElement::CalculateOnIntegrationPoints(
        const Variable<double>& rVariable,
        std::vector<double>& rValues,
        const ProcessInfo& rCurrentProcessInfo
    )
    {
        const auto& r_geometry = GetGeometry();
        const auto& r_integration_points = r_geometry.IntegrationPoints();

        if (rValues.size() != r_integration_points.size())
        {
            rValues.resize(r_integration_points.size());
        }

        //get properties
        const Vector& t = GetProperties()[TANGENTS];
        const double E = GetProperties()[YOUNG_MODULUS];
        const double A = GetProperties()[CROSS_AREA];
        const double prestress = GetProperties()[PRESTRESS_CAUCHY];

        if (rVariable == CABLE_STRESS)
        {
            for (IndexType point_number = 0; point_number < r_integration_points.size(); ++point_number) 
            {
                // get integration data
                const double& integration_weight = r_integration_points[point_number].Weight();
                const Matrix& r_DN_De   = r_geometry.ShapeFunctionLocalGradient(point_number);

                const array_1d<double, 3> actual_base_vector = GetActualBaseVector(r_DN_De);
                const double reference_a = norm_2(mReferenceBaseVector[point_number]);
                const double actual_a = norm_2(actual_base_vector);

                const double actual_aa = actual_a * actual_a;
                const double reference_aa = reference_a * reference_a;

                // green-lagrange strain
                const double e11_membrane = 0.5 * (actual_aa - reference_aa);

                // normal forcereference_aa
                double principal_stress = prestress * A + e11_membrane * A * E / inner_prod(mReferenceBaseVector[point_number],mReferenceBaseVector[point_number]); 

                rValues[point_number] = principal_stress;
            }   
        }
    }

    ///@}
    ///@name Dynamic Functions
    ///@{

    void IgaEdgeCableElement::GetValuesVector(
        Vector& rValues,
        int Step)
    {
        const SizeType number_of_control_points = GetGeometry().size();
        const SizeType mat_size = number_of_control_points * 3;

        if (rValues.size() != mat_size)
            rValues.resize(mat_size, false);

        for (IndexType i = 0; i < number_of_control_points; ++i)
        {
            const array_1d<double, 3 >& displacement = GetGeometry()[i].FastGetSolutionStepValue(DISPLACEMENT, Step);
            IndexType index = i * 3;

            rValues[index] = displacement[0];
            rValues[index + 1] = displacement[1];
            rValues[index + 2] = displacement[2];
        }
    }

    void IgaEdgeCableElement::GetFirstDerivativesVector(
        Vector& rValues,
        int Step)
    {
        const SizeType number_of_control_points = GetGeometry().size();
        const SizeType mat_size = number_of_control_points * 3;

        if (rValues.size() != mat_size)
            rValues.resize(mat_size, false);

        for (IndexType i = 0; i < number_of_control_points; ++i) {
            const array_1d<double, 3 >& velocity = GetGeometry()[i].FastGetSolutionStepValue(VELOCITY, Step);
            const IndexType index = i * 3;

            rValues[index] = velocity[0];
            rValues[index + 1] = velocity[1];
            rValues[index + 2] = velocity[2];
        }
    }

    void IgaEdgeCableElement::GetSecondDerivativesVector(
        Vector& rValues,
        int Step)
    {
        const SizeType number_of_control_points = GetGeometry().size();
        const SizeType mat_size = number_of_control_points * 3;

        if (rValues.size() != mat_size)
            rValues.resize(mat_size, false);

        for (IndexType i = 0; i < number_of_control_points; ++i) {
            const array_1d<double, 3 >& acceleration = GetGeometry()[i].FastGetSolutionStepValue(ACCELERATION, Step);
            const IndexType index = i * 3;

            rValues[index] = acceleration[0];
            rValues[index + 1] = acceleration[1];
            rValues[index + 2] = acceleration[2];
        }
    }

    void IgaEdgeCableElement::EquationIdVector(
        EquationIdVectorType& rResult,
        ProcessInfo& rCurrentProcessInfo
    )
    {
        KRATOS_TRY;

        const SizeType number_of_control_points = GetGeometry().size();

        if (rResult.size() != 3 * number_of_control_points)
            rResult.resize(3 * number_of_control_points, false);

        const IndexType pos = this->GetGeometry()[0].GetDofPosition(DISPLACEMENT_X);

        for (IndexType i = 0; i < number_of_control_points; ++i) {
            const IndexType index = i * 3;
            rResult[index]     = GetGeometry()[i].GetDof(DISPLACEMENT_X, pos).EquationId();
            rResult[index + 1] = GetGeometry()[i].GetDof(DISPLACEMENT_Y, pos + 1).EquationId();
            rResult[index + 2] = GetGeometry()[i].GetDof(DISPLACEMENT_Z, pos + 2).EquationId();
        }

        KRATOS_CATCH("")
    };

    void IgaEdgeCableElement::GetDofList(
        DofsVectorType& rElementalDofList,
        ProcessInfo& rCurrentProcessInfo
    )
    {
        KRATOS_TRY;

        const SizeType number_of_control_points = GetGeometry().size();

        rElementalDofList.resize(0);
        rElementalDofList.reserve(3 * number_of_control_points);

        for (IndexType i = 0; i < number_of_control_points; ++i) {
            rElementalDofList.push_back(GetGeometry()[i].pGetDof(DISPLACEMENT_X));
            rElementalDofList.push_back(GetGeometry()[i].pGetDof(DISPLACEMENT_Y));
            rElementalDofList.push_back(GetGeometry()[i].pGetDof(DISPLACEMENT_Z));
        }

        KRATOS_CATCH("")
    };

    ///@}
    ///@name Check
    ///@{

    int IgaEdgeCableElement::Check(const ProcessInfo& rCurrentProcessInfo)
    {
        // Verify that the constitutive law exists
        if (this->GetProperties().Has(CONSTITUTIVE_LAW) == false)
        {
            KRATOS_ERROR << "Constitutive law not provided for property " << this->GetProperties().Id() << std::endl;
        }
        else
        {
            // Verify that the constitutive law has the correct dimension
            KRATOS_ERROR_IF_NOT(this->GetProperties().Has(THICKNESS))
                << "THICKNESS not provided for element " << this->Id() << std::endl;

            // Check strain size
            KRATOS_ERROR_IF_NOT(this->GetProperties().GetValue(CONSTITUTIVE_LAW)->GetStrainSize() == 3)
                << "Wrong constitutive law used. This is a 2D element! Expected strain size is 3 (el id = ) "
                << this->Id() << std::endl;
        }

        return 0;
    }

    ///@}

} // Namespace Kratos
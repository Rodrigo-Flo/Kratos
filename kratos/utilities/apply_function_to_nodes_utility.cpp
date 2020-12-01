//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ \.
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Riccardo Rossi
//  Collaborator:    Vicente Mataix Ferrandiz
//

// System includes

// External includes

// Project includes
#include "utilities/apply_function_to_nodes_utility.h"

namespace Kratos
{

void ApplyFunctionToNodesUtility::ApplyFunction(
    const Variable<double>& rVariable,
    const double t
    )
{
    // The first node iterator
    const auto it_node_begin = mrNodes.begin();

    if(!mpFunction[0]->UseLocalSystem()) {
        #pragma omp parallel for
        for (int k = 0; k < static_cast<int>(mrNodes.size()); k++) {
            auto it_node = it_node_begin + k;
            const double value = mpFunction[OpenMPUtils::ThisThread()]->CallFunction(it_node->X(), it_node->Y(), it_node->Z(), t, it_node->X0(), it_node->Y0(), it_node->Z0());
            it_node->FastGetSolutionStepValue(rVariable) = value;
        }
    } else {
        #pragma omp parallel for
        for (int k = 0; k < static_cast<int>(mrNodes.size()); k++) {
            auto it_node = it_node_begin + k;
            const double value = mpFunction[OpenMPUtils::ThisThread()]->RotateAndCallFunction(it_node->X(), it_node->Y(), it_node->Z(), t, it_node->X0(), it_node->Y0(), it_node->Z0());
            it_node->FastGetSolutionStepValue(rVariable) = value;
        }
    }
}

/***********************************************************************************/
/***********************************************************************************/

std::vector<double> ApplyFunctionToNodesUtility::ReturnFunction(const double t)
{
    // The first node iterator
    const auto it_node_begin = mrNodes.begin();

    // The vector containing the values
    std::vector<double> values(mrNodes.size());

    if(!mpFunction[0]->UseLocalSystem()) {
        #pragma omp parallel for
        for (int k = 0; k < static_cast<int>(mrNodes.size()); k++) {
            auto it_node = it_node_begin + k;
            const double value = mpFunction[OpenMPUtils::ThisThread()]->CallFunction(it_node->X(), it_node->Y(), it_node->Z(), t, it_node->X0(), it_node->Y0(), it_node->Z0());
            values[k] = value;
        }
    } else {
        #pragma omp parallel for
        for (int k = 0; k < static_cast<int>(mrNodes.size()); k++) {
            auto it_node = it_node_begin + k;
            const double value = mpFunction[OpenMPUtils::ThisThread()]->RotateAndCallFunction(it_node->X(), it_node->Y(), it_node->Z(), t, it_node->X0(), it_node->Y0(), it_node->Z0());
            values[k] = value;
        }
    }

    return values;
}

} /// namespace Kratos
#include "custom_io/hdf5_element_solution_step_data_io.h"

#include "utilities/openmp_utils.h"
#include "custom_utilities/hdf5_data_set_partition_utility.h"
#include "includes/kratos_parameters.h"
#include "custom_utilities/registered_variable_lookup.h"

namespace Kratos
{
namespace HDF5
{
namespace
{
template <class TVariableType, class TFileDataType>
void SetDataBuffer(TVariableType const& rVariable,
                   std::vector<ElementType*> const& rElements,
                   Vector<TFileDataType>& rData
                   );

template <class TVariableType, class TFileDataType>
void SetElementSolutionStepData(TVariableType const& rVariable,
                              Vector<TFileDataType> const& rData,
                              std::vector<ElementType*>& rElements
                              );

template <typename TVariable>
class WriteVariableFunctor
{
public:
    void operator()(TVariable const& rVariable,
                    std::vector<ElementType*>& rElements,
                    File& rFile,
                    std::string const& rPrefix,
                    WriteInfo& rInfo)
    {
        Vector<typename TVariable::Type> data;
        SetDataBuffer(rVariable, rElements, data);
        rFile.WriteDataSet(rPrefix + "/ElementResults/" + rVariable.Name(), data, rInfo);
    }
};

template <typename TVariable>
class ReadVariableFunctor
{
public:
    void operator()(TVariable const& rVariable,
                    std::vector<ElementType*>& rElements,
                    File& rFile,
                    std::string const& rPrefix,
                    unsigned StartIndex,
                    unsigned BlockSize)
    {
        Vector<typename TVariable::Type> data;
        rFile.ReadDataSet(rPrefix + "/ElementResults/" + rVariable.Name(), data,
                          StartIndex, BlockSize);
        SetElementSolutionStepData(rVariable, data, rElements);
    }
};

std::vector<ElementType*> GetElementReferences(ElementsContainerType const& rElements )
{
    std::vector<ElementType*> rElementReferences;
    rElementReferences.resize(rElements.size());

    #pragma omp parallel for
        for (int i = 0; i < rElements.size(); ++i)
        {
            auto it = rElements.begin() + i;
            rElementReferences[i] = (&(*it));
        }
    
    return rElementReferences;
}
} // unnamed namespace

ElementSolutionStepDataIO::ElementSolutionStepDataIO(Parameters Settings, File::Pointer pFile)
    : mpFile(pFile)
{
    KRATOS_TRY;

    Parameters default_params(R"(
        {
            "prefix": "",
            "list_of_variables": []
        })");

    Settings.ValidateAndAssignDefaults(default_params);

    mPrefix = Settings["prefix"].GetString();

    mVariableNames.resize(Settings["list_of_variables"].size());
    for (unsigned i = 0; i < mVariableNames.size(); ++i)
        mVariableNames[i] = Settings["list_of_variables"].GetArrayItem(i).GetString();

    KRATOS_CATCH("");
}

void ElementSolutionStepDataIO::WriteElementResults(ElementsContainerType const& rElements)
{
    KRATOS_TRY;

    if (mVariableNames.size() == 0)
        return;

    WriteInfo info;
    std::vector<ElementType*> local_elements = GetElementReferences(rElements);

    // Write each variable.
    for (const std::string& r_variable_name : mVariableNames)
        RegisteredVariableLookup<Variable<array_1d<double, 3>>,
                                 VariableComponent<VectorComponentAdaptor<array_1d<double, 3>>>,
                                 Variable<double>, Variable<int>>(r_variable_name)
            .Execute<WriteVariableFunctor>(local_elements, *mpFile, mPrefix, info);

    // Write block partition.
    WritePartitionTable(*mpFile, mPrefix + "/ElementResults", info);

    KRATOS_CATCH("");
}

void ElementSolutionStepDataIO::ReadElementResults(ElementsContainerType& rElements)
{
    KRATOS_TRY;

    if (mVariableNames.size() == 0)
        return;

    std::vector<ElementType*> local_elements = GetElementReferences(rElements);
    unsigned start_index, block_size;
    std::tie(start_index, block_size) = StartIndexAndBlockSize(*mpFile, mPrefix + "/ElementResults");

    // Read local data for each variable.
    for (const std::string& r_variable_name : mVariableNames)
        RegisteredVariableLookup<Variable<array_1d<double, 3>>,
                                 VariableComponent<VectorComponentAdaptor<array_1d<double, 3>>>,
                                 Variable<double>, Variable<int>>(r_variable_name)
            .Execute<ReadVariableFunctor>(local_elements, *mpFile, mPrefix,
                                          start_index, block_size);

    KRATOS_CATCH("");
}

namespace
{
template <class TVariableType, class TFileDataType>
void SetDataBuffer(TVariableType const& rVariable,
                   std::vector<ElementType*> const& rElements,
                   Vector<TFileDataType>& rData
                   )
{
    KRATOS_TRY;

    rData.resize(rElements.size(), false);

#pragma omp parallel for
    for (int i = 0; i < rElements.size(); ++i)
            rData[i] = rElements[i]->GetValue(rVariable);

    KRATOS_CATCH("");
}

template <class TVariableType, class TFileDataType>
void SetElementSolutionStepData(TVariableType const& rVariable,
                              Vector<TFileDataType> const& rData,
                              std::vector<ElementType*>& rElements
                              )
{
    KRATOS_TRY;

    KRATOS_ERROR_IF(rData.size() != rElements.size())
        << "File data block size (" << rData.size()
        << ") is not equal to number of nodes (" << rElements.size() << ")." << std::endl;

#pragma omp parallel for
    for (int i = 0; i < rElements.size(); ++i)
        rElements[i]->SetValue(rVariable, rData[i]);

    KRATOS_CATCH("");
}
} // unnamed namespace.
} // namespace HDF5.
} // namespace Kratos.
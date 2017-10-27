//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Ruben Zorrilla
//

// System includes

// External includes

// Project includes
#include "modified_shape_functions/modified_shape_functions.h"

namespace Kratos
{
    
    /// ModifiedShapeFunctions implementation
    /// Default constructor
    ModifiedShapeFunctions::ModifiedShapeFunctions(const GeometryPointerType pInputGeometry, const Vector& rNodalDistances) :
        mpInputGeometry(pInputGeometry), mrNodalDistances(rNodalDistances) {
    };

    /// Destructor
    ModifiedShapeFunctions::~ModifiedShapeFunctions() {};

    /// Turn back information as a string.
    std::string ModifiedShapeFunctions::Info() const {
        return "Modified shape functions computation base class.";
    };
    
    /// Print information about this object.
    void ModifiedShapeFunctions::PrintInfo(std::ostream& rOStream) const {
        rOStream << "Modified shape functions computation base class.";
    };
    
    /// Print object's data.
    void ModifiedShapeFunctions::PrintData(std::ostream& rOStream) const {
        const GeometryPointerType p_geometry = this->GetInputGeometry();
        const Vector nodal_distances = this->GetNodalDistances();
        rOStream << "Modified shape functions computation base class:\n";
        rOStream << "\tGeometry type: " << (*p_geometry).Info() << "\n";
        std::stringstream distances_buffer;
        for (unsigned int i = 0; i < nodal_distances.size(); ++i) {
            distances_buffer << std::to_string(nodal_distances(i)) << " ";
        }
        rOStream << "\tDistance values: " << distances_buffer.str();
    };
    
    // Returns the input original geometry.
    const ModifiedShapeFunctions::GeometryPointerType ModifiedShapeFunctions::GetInputGeometry() const {
        return mpInputGeometry;
    };
    
    // Returns the nodal distances vector.
    const Vector& ModifiedShapeFunctions::GetNodalDistances() const {
        return mrNodalDistances;
    };

    // Internally computes the splitting pattern and returns all the shape function values for the positive side.
    void ModifiedShapeFunctions::GetPositiveSideShapeFunctionsAndGradientsValues(Matrix &rPositiveSideShapeFunctionsValues,
                                                                                 std::vector<Matrix> &rPositiveSideShapeFunctionsGradientsValues,
                                                                                 Vector &rPositiveSideWeightsValues,
                                                                                 const IntegrationMethodType IntegrationMethod) {
        KRATOS_ERROR << "Calling the base class GetShapeFunctionsAndGradientsValuesPositiveSide method. Call the specific geometry one.";
    };

    // Internally computes the splitting pattern and returns all the shape function values for the negative side.
    void ModifiedShapeFunctions::GetNegativeSideShapeFunctionsAndGradientsValues(Matrix &rNegativeSideShapeFunctionsValues,
                                                                                 std::vector<Matrix> &rNegativeSideShapeFunctionsGradientsValues,
                                                                                 Vector &rNegativeSideWeightsValues,
                                                                                 const IntegrationMethodType IntegrationMethod) {
        KRATOS_ERROR << "Calling the base class GetShapeFunctionsAndGradientsValuesNegativeSide method. Call the specific geometry one.";
    };

    // Internally computes the splitting pattern and returns all the shape function values for the positive interface side.
    void ModifiedShapeFunctions::GetInterfacePositiveSideShapeFunctionsAndGradientsValues(Matrix &rInterfacePositiveSideShapeFunctionsValues,
                                                                                 std::vector<Matrix> &rPositiveInterfaceSideShapeFunctionsGradientsValues,
                                                                                 Vector &rPositiveInterfaceSideWeightsValues,
                                                                                 const IntegrationMethodType IntegrationMethod) {
        KRATOS_ERROR << "Calling the base class GetInterfaceShapeFunctionsAndGradientsValuesPositiveSide method. Call the specific geometry one.";
    };

    // Internally computes the splitting pattern and returns all the shape function values for the negative interface side.
    void ModifiedShapeFunctions::GetInterfaceNegativeSideShapeFunctionsAndGradientsValues(Matrix &rInterfaceNegativeSideShapeFunctionsValues,
                                                                                 std::vector<Matrix> &rInterfaceNegativeSideShapeFunctionsGradientsValues,
                                                                                 Vector &rInterfaceNegativeSideWeightsValues,
                                                                                 const IntegrationMethodType IntegrationMethod) {
        KRATOS_ERROR << "Calling the base class GetInterfaceShapeFunctionsAndGradientsValuesNegativeSide method. Call the specific geometry one.";
    };
    
    // Sets the condensation matrix to transform the subdivsion values to entire element ones.
    void ModifiedShapeFunctions::SetIntersectionPointsCondensationMatrix(Matrix& rIntPointCondMatrix,
                                                                         const int EdgeNodeI[],
                                                                         const int EdgeNodeJ[],
                                                                         const int SplitEdges[],
                                                                         const unsigned int SplitEdgesSize) {

        // Initialize intersection points condensation matrix
        const unsigned int nedges = mpInputGeometry->EdgesNumber();
        const unsigned int nnodes = mpInputGeometry->PointsNumber();

        rIntPointCondMatrix = ZeroMatrix(SplitEdgesSize, nnodes);

        // Fill the original geometry points main diagonal
        for (unsigned int i = 0; i < nnodes; ++i) {
            rIntPointCondMatrix(i,i) = 1.0;
        }

        // Compute the intersection points contributions
        unsigned int row = nnodes;
        for (unsigned int idedge = 0; idedge < nedges; ++idedge) {
            // Check if the edge has an intersection point
            if (SplitEdges[nnodes+idedge] != -1) {
                // Get the nodes that compose the edge
                const unsigned int edge_node_i = EdgeNodeI[idedge];
                const unsigned int edge_node_j = EdgeNodeJ[idedge];

                // Compute the relative coordinate of the intersection point over the edge
                const double aux_node_rel_location = std::abs (mrNodalDistances(edge_node_i)/(mrNodalDistances(edge_node_j)-mrNodalDistances(edge_node_i)));

                // Store the relative coordinate values as the original geometry nodes sh. function value in the intersections
                rIntPointCondMatrix(row, edge_node_i) = aux_node_rel_location;
                rIntPointCondMatrix(row, edge_node_j) = 1.0 - aux_node_rel_location;
            }
            row++;
        }
    }

    // Given the subdivision pattern of either the positive or negative side, computes the shape function values. 
    void ModifiedShapeFunctions::ComputeValuesOnOneSide(Matrix &rShapeFunctionsValues,
                                                        std::vector<Matrix> &rShapeFunctionsGradientsValues,
                                                        Vector &rWeightsValues,
                                                        const std::vector<IndexedPointGeometryPointerType> &rSubdivisionsVector,
                                                        const Matrix &rPmatrix,
                                                        const IntegrationMethodType IntegrationMethod) {

        // Set some auxiliar constants
        GeometryPointerType p_input_geometry = this->GetInputGeometry();
        const unsigned int n_edges_global = p_input_geometry->EdgesNumber();       // Number of edges in original geometry
        const unsigned int n_nodes_global = p_input_geometry->PointsNumber();      // Number of nodes in original geometry
        const unsigned int split_edges_size = n_edges_global + n_nodes_global;     // Split edges vector size 

        const unsigned int n_subdivision = rSubdivisionsVector.size();                                         // Number of positive or negative subdivisions
        const unsigned int n_dim = (*rSubdivisionsVector[0]).Dimension();                                      // Number of dimensions 
        const unsigned int n_nodes = (*rSubdivisionsVector[0]).PointsNumber();                                 // Number of nodes per subdivision
        const unsigned int n_int_pts = (*rSubdivisionsVector[0]).IntegrationPointsNumber(IntegrationMethod);   // Number of Gauss pts. per subdivision
 
        // Resize the shape function values matrix
        const unsigned int n_total_int_pts = n_subdivision * n_int_pts;

        if (rShapeFunctionsValues.size1() != n_total_int_pts) {
            rShapeFunctionsValues.resize(n_total_int_pts, n_nodes, false);
        } else if (rShapeFunctionsValues.size2() != n_nodes) {
            rShapeFunctionsValues.resize(n_total_int_pts, n_nodes, false);
        }

        // Resize the weights vector
        if (rWeightsValues.size() != n_total_int_pts) {
            rWeightsValues.resize(n_total_int_pts, false);
        }

        // Compute each Gauss pt. shape functions values
        for (unsigned int i_subdivision = 0; i_subdivision < n_subdivision; ++i_subdivision) {
            
            const IndexedPointGeometryType& r_subdivision_geom = *rSubdivisionsVector[i_subdivision];
            
            // Get the subdivision shape function values
            const Matrix subdivision_sh_func_values = r_subdivision_geom.ShapeFunctionsValues(IntegrationMethod);
            ShapeFunctionsGradientsType subdivision_sh_func_gradients_values;
            r_subdivision_geom.ShapeFunctionsIntegrationPointsGradients(subdivision_sh_func_gradients_values, IntegrationMethod);

            // Get the subdivision Jacobian values on all Gauss pts.
            Vector subdivision_jacobians_values;
            r_subdivision_geom.DeterminantOfJacobian(subdivision_jacobians_values, IntegrationMethod);

            // Get the subdivision Gauss pts. (x_coord, y_coord, z_coord, weight)
            const IntegrationPointsArrayType subdivision_gauss_points = r_subdivision_geom.IntegrationPoints(IntegrationMethod);

            // Apply the original nodes condensation
            for (unsigned int i_gauss = 0; i_gauss < n_int_pts; ++i_gauss) {

                // Store the Gauss pts. weights values
                rWeightsValues(i_subdivision*n_int_pts + i_gauss) = subdivision_jacobians_values(i_gauss) * subdivision_gauss_points[i_gauss].Weight();

                // Condense the shape function local values to obtain the original nodes ones
                Vector sh_func_vect = ZeroVector(split_edges_size);
                for (unsigned int i = 0; i < n_nodes; ++i) {
                    sh_func_vect(r_subdivision_geom[i].Id()) = subdivision_sh_func_values(i_gauss, i); 
                }
                
                const Vector condensed_sh_func_values = prod(sh_func_vect, rPmatrix);
                
                for (unsigned int i = 0; i < n_nodes; ++i) {
                    rShapeFunctionsValues(i_subdivision*n_int_pts + i_gauss, i) = condensed_sh_func_values(i);
                }
                
                // Condense the shape function gradients local values to obtain the original nodes ones
                Matrix aux_gauss_gradients = ZeroMatrix(n_nodes, n_dim);

                Matrix sh_func_gradients_mat = ZeroMatrix(n_dim, split_edges_size);
                for (unsigned int dim = 0; dim < n_dim; ++dim ) {
                    for (unsigned int i_node = 0; i_node < n_nodes; ++i_node) {
                        sh_func_gradients_mat(dim, r_subdivision_geom[i_node].Id()) = subdivision_sh_func_gradients_values[i_gauss](i_node, dim); 
                    }
                }

                rShapeFunctionsGradientsValues.push_back(trans(prod(sh_func_gradients_mat, rPmatrix)));
            }
        }
    };

    // Given the subdivision pattern of either the positive or negative interface side, computes the shape function values. 
    void ModifiedShapeFunctions::ComputeInterfaceValuesOnOneSide(Matrix &rInterfaceShapeFunctionsValues,
                                                                 std::vector<Matrix> &rInterfaceShapeFunctionsGradientsValues,
                                                                 Vector &rInterfaceWeightsValues,
                                                                 const std::vector<IndexedPointGeometryPointerType> &rInterfacesVector,
                                                                 const Matrix &rPmatrix,
                                                                 const IntegrationMethodType IntegrationMethod) {
        // Set some auxiliar variables
        const unsigned int n_interfaces = rInterfacesVector.size();                                          // Number of positive or negative subdivisions
        const unsigned int n_dim = (*rInterfacesVector[0]).Dimension();                                      // Number of dimensions 
        const unsigned int n_nodes = (*rInterfacesVector[0]).PointsNumber();                                 // Number of nodes per subdivision
        const unsigned int n_int_pts = (*rInterfacesVector[0]).IntegrationPointsNumber(IntegrationMethod);   // Number of Gauss pts. per subdivision
        GeometryPointerType p_input_geometry = this->GetInputGeometry();                                     // Pointer to the input geometry

        // Resize the shape function values matrix
        const unsigned int n_total_int_pts = n_interfaces * n_int_pts;

        if (rInterfaceShapeFunctionsValues.size1() != n_total_int_pts) {
            rInterfaceShapeFunctionsValues.resize(n_total_int_pts, n_nodes, false);
        } else if (rInterfaceShapeFunctionsValues.size2() != n_nodes) {
            rInterfaceShapeFunctionsValues.resize(n_total_int_pts, n_nodes, false);
        }

        // Resize the weights vector
        if (rInterfaceWeightsValues.size() != n_total_int_pts) {
            rInterfaceWeightsValues.resize(n_total_int_pts, false);
        }

        // Compute each Gauss pt. shape functions values
        for (unsigned int i_interface = 0; i_interface < n_interfaces; ++i_interface) {
            
            const IndexedPointGeometryType& r_interface_geom = *rInterfacesVector[i_interface];

            // Get the intersection Gauss points coordinates
            IntegrationPointsArrayType r_interface_gauss_pts = r_interface_geom.IntegrationPoints(IntegrationMethod);

            // Get the global coordinates of the intersection Gauss pts.
            std::vector < CoordinatesArrayType > r_interface_gauss_pts_global_coords;
            for (unsigned int i_gauss=0; i_gauss<n_int_pts; ++i_gauss) {
                CoordinatesArrayType global_coordinates = ZeroVector(3);
                r_interface_geom.GlobalCoordinates(global_coordinates, r_interface_gauss_pts[i_gauss].Coordinates());
                r_interface_gauss_pts_global_coords.push_back(global_coordinates);
            }

            // Get the original local coordinates of the intersection Gauss pts.
            std::vector < CoordinatesArrayType > r_interface_gauss_pts_local_coords_input_geom;
            for (unsigned int i_gauss=0; i_gauss<n_int_pts; ++i_gauss) {
                CoordinatesArrayType local_coordinates = ZeroVector(3);
                local_coordinates = p_input_geometry->PointLocalCoordinates(local_coordinates, r_interface_gauss_pts_global_coords[i_gauss]);
                KRATOS_WATCH("")
                KRATOS_WATCH(r_interface_gauss_pts_global_coords[i_gauss])
                KRATOS_WATCH(local_coordinates)
                r_interface_gauss_pts_local_coords_input_geom.push_back(local_coordinates);
            }
            
            // // Get the subdivision shape function values
            // const Matrix interface_sh_func_values = r_interface_geom.ShapeFunctionsValues(IntegrationMethod);
            // ShapeFunctionsGradientsType interface_sh_func_gradients_values;
            // r_interface_geom.ShapeFunctionsIntegrationPointsGradients(interface_sh_func_gradients_values, IntegrationMethod);

            // // Get the subdivision Jacobian values on all Gauss pts.
            // Vector interface_jacobians_values;
            // r_interface_geom.DeterminantOfJacobian(interface_jacobians_values, IntegrationMethod);

            // // Get the subdivision Gauss pts. (x_coord, y_coord, z_coord, weight)
            // const IntegrationPointsArrayType interface_gauss_points = r_interface_geom.IntegrationPoints(IntegrationMethod);

            // // Apply the original nodes condensation
            // for (unsigned int i_gauss = 0; i_gauss < n_int_pts; ++i_gauss) {

            //     // Store the Gauss pts. weights values
            //     rWeightsValues(i_interface*n_int_pts + i_gauss) = interface_jacobians_values(i_gauss) * interface_gauss_points[i_gauss].Weight();

            //     // // Condense the shape function local values to obtain the original nodes ones
            //     // Vector sh_func_vect = ZeroVector(split_edges_size);
            //     // for (unsigned int i = 0; i < n_nodes; ++i) {
            //     //     sh_func_vect(r_interface_geom[i].Id()) = interface_sh_func_values(i_gauss, i); 
            //     // }
                
            //     // const Vector condensed_sh_func_values = prod(sh_func_vect, rPmatrix);
                
            //     // for (unsigned int i = 0; i < n_nodes; ++i) {
            //     //     rShapeFunctionsValues(i_interface*n_int_pts + i_gauss, i) = condensed_sh_func_values(i);
            //     // }
                
            //     // Condense the shape function gradients local values to obtain the original nodes ones
            //     Matrix aux_gauss_gradients = ZeroMatrix(n_nodes, n_dim);

            //     Matrix sh_func_gradients_mat = ZeroMatrix(n_dim, split_edges_size);
            //     for (unsigned int dim = 0; dim < n_dim; ++dim ) {
            //         for (unsigned int i_node = 0; i_node < n_nodes; ++i_node) {
            //             sh_func_gradients_mat(dim, r_interface_geom[i_node].Id()) = interface_sh_func_gradients_values[i_gauss](i_node, dim); 
            //         }
            //     }

            //     rShapeFunctionsGradientsValues.push_back(trans(prod(sh_func_gradients_mat, rPmatrix)));
            // }
        }
 
    };


}; // namespace Kratos

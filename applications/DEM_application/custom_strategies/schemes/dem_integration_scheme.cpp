//        
// Author: Miguel AngelCeligueta, maceli@cimne.upc.edu
//

#include "custom_utilities/GeometryFunctions.h"
#include "custom_elements/cluster3D.h"
#include "dem_integration_scheme.h"
#include "DEM_application_variables.h"


namespace Kratos {

    DEMIntegrationScheme::DEMIntegrationScheme(){}
    DEMIntegrationScheme::~DEMIntegrationScheme(){}
    
    void DEMIntegrationScheme::SetIntegrationSchemeInProperties(Properties::Pointer pProp, bool verbose) const {
        //if(verbose) std::cout << "\nAssigning DEMDiscontinuumConstitutiveLaw to properties " << pProp->Id() << std::endl;
        pProp->SetValue(DEM_TRANSLATIONAL_INTEGRATION_SCHEME_POINTER, this->CloneShared());
        pProp->SetValue(DEM_ROTATIONAL_INTEGRATION_SCHEME_POINTER, this->CloneShared());
    }
    
    void DEMIntegrationScheme::Move(Node<3> & i, const double delta_t, const double force_reduction_factor, const int StepFlag) {
        if (i.Is(DEMFlags::BELONGS_TO_A_CLUSTER)) return;
        CalculateTranslationalMotionOfNode(i, delta_t, force_reduction_factor, StepFlag);
    }
    
    void DEMIntegrationScheme::Rotate(Node<3> & i, const double delta_t, const double force_reduction_factor, const int StepFlag) {
        if (i.Is(DEMFlags::BELONGS_TO_A_CLUSTER)) return;
        CalculateRotationalMotionOfNode(i, delta_t, force_reduction_factor, StepFlag);
    }
    
    void DEMIntegrationScheme::MoveCluster(Cluster3D* cluster_element, Node<3> & i, const double delta_t, const double force_reduction_factor, const int StepFlag) {
        CalculateTranslationalMotionOfNode(i, delta_t, force_reduction_factor, StepFlag);   
        cluster_element->UpdateLinearDisplacementAndVelocityOfSpheres();
    }
    
    void DEMIntegrationScheme::RotateCluster(Cluster3D* cluster_element, Node<3> & i, const double delta_t, const double force_reduction_factor, const int StepFlag) {
        CalculateRotationalMotionOfClusterNode(i, delta_t, force_reduction_factor, StepFlag);                
        cluster_element->UpdateAngularDisplacementAndVelocityOfSpheres();
    }
    
    void DEMIntegrationScheme::CalculateTranslationalMotionOfNode(Node<3> & i, const double delta_t, const double force_reduction_factor, const int StepFlag) {
        array_1d<double, 3 >& vel = i.FastGetSolutionStepValue(VELOCITY);
        array_1d<double, 3 >& displ = i.FastGetSolutionStepValue(DISPLACEMENT);
        array_1d<double, 3 >& delta_displ = i.FastGetSolutionStepValue(DELTA_DISPLACEMENT);
        array_1d<double, 3 >& coor = i.Coordinates();
        array_1d<double, 3 >& initial_coor = i.GetInitialPosition();
        array_1d<double, 3 >& force = i.FastGetSolutionStepValue(TOTAL_FORCES);
        
        #ifdef KRATOS_DEBUG
        DemDebugFunctions::CheckIfNan(force, "NAN in Force in Integration Scheme");
        #endif  
        
        double mass = i.FastGetSolutionStepValue(NODAL_MASS);                   

        bool Fix_vel[3] = {false, false, false};

        Fix_vel[0] = i.Is(DEMFlags::FIXED_VEL_X);
        Fix_vel[1] = i.Is(DEMFlags::FIXED_VEL_Y);
        Fix_vel[2] = i.Is(DEMFlags::FIXED_VEL_Z);

        UpdateTranslationalVariables(StepFlag, i, coor, displ, delta_displ, vel, initial_coor, force, force_reduction_factor, mass, delta_t, Fix_vel);       
    }
    
    void DEMIntegrationScheme::CalculateRotationalMotionOfNode(Node<3> & i, const double delta_t, const double moment_reduction_factor, const int StepFlag) {
    
        double moment_of_inertia               = i.FastGetSolutionStepValue(PARTICLE_MOMENT_OF_INERTIA);
        array_1d<double, 3 >& angular_velocity = i.FastGetSolutionStepValue(ANGULAR_VELOCITY);
        array_1d<double, 3 >& torque           = i.FastGetSolutionStepValue(PARTICLE_MOMENT);
        array_1d<double, 3 >& rotated_angle    = i.FastGetSolutionStepValue(PARTICLE_ROTATION_ANGLE);
        array_1d<double, 3 >& delta_rotation   = i.FastGetSolutionStepValue(DELTA_ROTATION);
        
        #ifdef KRATOS_DEBUG
        DemDebugFunctions::CheckIfNan(torque, "NAN in Torque in Integration Scheme");
        #endif

        bool Fix_Ang_vel[3] = {false, false, false};
        Fix_Ang_vel[0] = i.Is(DEMFlags::FIXED_ANG_VEL_X);
        Fix_Ang_vel[1] = i.Is(DEMFlags::FIXED_ANG_VEL_Y);
        Fix_Ang_vel[2] = i.Is(DEMFlags::FIXED_ANG_VEL_Z);
        
        CalculateNewRotationalVariables(StepFlag, i, moment_of_inertia, angular_velocity, torque, moment_reduction_factor, rotated_angle, delta_rotation, delta_t, Fix_Ang_vel);

//         array_1d<double, 3 > angular_acceleration;                    
//         CalculateLocalAngularAcceleration(i, moment_of_inertia, torque, force_reduction_factor,angular_acceleration);

//         UpdateRotationalVariables(StepFlag, i, rotated_angle, delta_rotation, angular_velocity, angular_acceleration, delta_t, Fix_Ang_vel);
    }
    
    void DEMIntegrationScheme::CalculateRotationalMotionOfClusterNode(Node<3> & i, const double delta_t, const double moment_reduction_factor, const int StepFlag) {
        
        array_1d<double, 3 >& moments_of_inertia = i.FastGetSolutionStepValue(PRINCIPAL_MOMENTS_OF_INERTIA);
        array_1d<double, 3 >& angular_velocity   = i.FastGetSolutionStepValue(ANGULAR_VELOCITY);
        array_1d<double, 3 >& torque             = i.FastGetSolutionStepValue(PARTICLE_MOMENT);
        array_1d<double, 3 >& rotated_angle      = i.FastGetSolutionStepValue(PARTICLE_ROTATION_ANGLE);
        array_1d<double, 3 >& delta_rotation     = i.FastGetSolutionStepValue(DELTA_ROTATION);
        Quaternion<double  >& Orientation        = i.FastGetSolutionStepValue(ORIENTATION);
        
        #ifdef KRATOS_DEBUG
        DemDebugFunctions::CheckIfNan(torque, "NAN in Torque in Integration Scheme");
        #endif

        bool Fix_Ang_vel[3] = {false, false, false};

        Fix_Ang_vel[0] = i.Is(DEMFlags::FIXED_ANG_VEL_X);
        Fix_Ang_vel[1] = i.Is(DEMFlags::FIXED_ANG_VEL_Y);
        Fix_Ang_vel[2] = i.Is(DEMFlags::FIXED_ANG_VEL_Z);
        
        CalculateNewRotationalVariables(StepFlag, i, moments_of_inertia, angular_velocity, torque, moment_reduction_factor, rotated_angle, delta_rotation, Orientation, delta_t, Fix_Ang_vel);
                
//         int type = 1; // 0 for RK algorithm, 1 for Zhao algorithm and 2 for classical method
// 
//         if (type == 0) {
//             
//             array_1d<double, 3 > & angular_momentum       = i.FastGetSolutionStepValue(ANGULAR_MOMENTUM);
//             array_1d<double, 3 > & local_angular_velocity = i.FastGetSolutionStepValue(LOCAL_ANGULAR_VELOCITY);
// 
//             array_1d<double, 3 > angular_momentum_aux;
//             angular_momentum_aux[0] = 0.0;
//             angular_momentum_aux[1] = 0.0;
//             angular_momentum_aux[2] = 0.0;
// 
//             if (Fix_Ang_vel[0] == true || Fix_Ang_vel[1] == true || Fix_Ang_vel[2] == true) {
//                 double LocalTensor[3][3];
//                 double GlobalTensor[3][3];
//                 GeometryFunctions::ConstructLocalTensor(moments_of_inertia, LocalTensor);
//                 GeometryFunctions::QuaternionTensorLocal2Global(Orientation, LocalTensor, GlobalTensor);
//                 GeometryFunctions::ProductMatrix3X3Vector3X1(GlobalTensor, angular_velocity, angular_momentum_aux);
//             }
// 
//             double dt = 0.0;
// 
//             if (StepFlag == 1 || StepFlag == 2) {dt = 0.5*delta_t;}
// 
//             else {dt = delta_t;}
// 
//             for (int j = 0; j < 3; j++) {
//                 if (Fix_Ang_vel[j] == false){
//                     angular_momentum[j] += moment_reduction_factor * torque[j] * dt;
//                 }
//                 else {
//                     angular_momentum[j] = angular_momentum_aux[j];
//                 }
//             }
// 
//             CalculateAngularVelocityRK(Orientation, moments_of_inertia, angular_momentum, angular_velocity, dt, Fix_Ang_vel);
//             UpdateRotationalVariablesOfCluster(i, moments_of_inertia, rotated_angle, delta_rotation, Orientation, angular_momentum, angular_velocity, dt, Fix_Ang_vel);
//             GeometryFunctions::QuaternionVectorGlobal2Local(Orientation, angular_velocity, local_angular_velocity);
// 
//         }//if type == 0
                
//         if (type == 1) {
//             array_1d<double, 3 > local_angular_acceleration, local_torque, quarter_local_angular_velocity, quarter_angular_velocity, AuxAngularVelocity;
//             array_1d<double, 3 > & local_angular_velocity = i.FastGetSolutionStepValue(LOCAL_ANGULAR_VELOCITY);
//             Quaternion<double  > & AuxOrientation = i.FastGetSolutionStepValue(AUX_ORIENTATION);
//             array_1d<double, 3 > & LocalAuxAngularVelocity = i.FastGetSolutionStepValue(LOCAL_AUX_ANGULAR_VELOCITY);
// 
//             if (StepFlag != 1 && StepFlag != 2) {
//                 //Angular velocity and torques are saved in the local framework:
//                 GeometryFunctions::QuaternionVectorGlobal2Local(Orientation, torque, local_torque);
//                 CalculateLocalAngularAccelerationByEulerEquations(i,local_angular_velocity,moments_of_inertia,local_torque,moment_reduction_factor,local_angular_acceleration);
//                     
//                 quarter_local_angular_velocity = local_angular_velocity + 0.25 * local_angular_acceleration * delta_t;
//                 LocalAuxAngularVelocity        = local_angular_velocity + 0.5  * local_angular_acceleration * delta_t;
//                     
//                 GeometryFunctions::QuaternionVectorLocal2Global(Orientation, quarter_local_angular_velocity, quarter_angular_velocity);
//                         
//                 array_1d<double, 3 > rotation_aux = 0.5 * quarter_angular_velocity * delta_t;
//                 GeometryFunctions::UpdateOrientation(Orientation, AuxOrientation, rotation_aux);
// 
//                 GeometryFunctions::QuaternionVectorLocal2Global(AuxOrientation, LocalAuxAngularVelocity, AuxAngularVelocity);
//                 UpdateRotationalVariables(i, rotated_angle, delta_rotation, AuxAngularVelocity, delta_t, Fix_Ang_vel);
//                 GeometryFunctions::UpdateOrientation(Orientation, delta_rotation);
// 
//                 //Angular velocity and torques are saved in the local framework:
//                 GeometryFunctions::QuaternionVectorGlobal2Local(AuxOrientation, torque, local_torque);
//                 CalculateLocalAngularAccelerationByEulerEquations(i,LocalAuxAngularVelocity,moments_of_inertia,local_torque, moment_reduction_factor,local_angular_acceleration);
// 
//                 local_angular_velocity += local_angular_acceleration * delta_t;
//                 GeometryFunctions::QuaternionVectorLocal2Global(Orientation, local_angular_velocity, angular_velocity);
// 
//                 GeometryFunctions::QuaternionVectorLocal2Global(AuxOrientation, LocalAuxAngularVelocity, AuxAngularVelocity);
//                 UpdateRotationalVariables(i, rotated_angle, delta_rotation, AuxAngularVelocity, delta_t, Fix_Ang_vel);
//                 GeometryFunctions::UpdateOrientation(Orientation, delta_rotation);
//             }
// 
//             if (StepFlag == 1) { //PREDICT
//                 //Angular velocity and torques are saved in the local framework:
//                 GeometryFunctions::QuaternionVectorGlobal2Local(Orientation, torque, local_torque);
//                 CalculateLocalAngularAccelerationByEulerEquations(i,local_angular_velocity,moments_of_inertia,local_torque,moment_reduction_factor,local_angular_acceleration);
// 
//                 quarter_local_angular_velocity = local_angular_velocity + 0.25 * local_angular_acceleration * delta_t;
//                 LocalAuxAngularVelocity        = local_angular_velocity + 0.5  * local_angular_acceleration * delta_t;
// 
//                 GeometryFunctions::QuaternionVectorLocal2Global(Orientation, quarter_local_angular_velocity, quarter_angular_velocity);
// 
//                 array_1d<double, 3 > rotation_aux = 0.5 * quarter_angular_velocity * delta_t;
//                 GeometryFunctions::UpdateOrientation(Orientation, AuxOrientation, rotation_aux);
// 
//                 GeometryFunctions::QuaternionVectorLocal2Global(AuxOrientation, LocalAuxAngularVelocity, AuxAngularVelocity);
//                 UpdateRotationalVariables(i, rotated_angle, delta_rotation, AuxAngularVelocity, delta_t, Fix_Ang_vel);
//                 GeometryFunctions::UpdateOrientation(Orientation, delta_rotation);
//             }//if StepFlag == 1
//                     
//             if (StepFlag == 2) { //CORRECT
//                 //Angular velocity and torques are saved in the local framework:
//                 GeometryFunctions::QuaternionVectorGlobal2Local(AuxOrientation, torque, local_torque);
//                 CalculateLocalAngularAccelerationByEulerEquations(i,LocalAuxAngularVelocity,moments_of_inertia,local_torque, moment_reduction_factor,local_angular_acceleration);
//                         
//                 local_angular_velocity += local_angular_acceleration * delta_t;
//                 GeometryFunctions::QuaternionVectorLocal2Global(Orientation, local_angular_velocity, angular_velocity);
//                         
//                 GeometryFunctions::QuaternionVectorLocal2Global(AuxOrientation, LocalAuxAngularVelocity, AuxAngularVelocity);
//                 UpdateRotationalVariables(i, rotated_angle, delta_rotation, AuxAngularVelocity, delta_t, Fix_Ang_vel);
//                 GeometryFunctions::UpdateOrientation(Orientation, delta_rotation);
//             }//if StepFlag == 2
//         }//if type == 1
                
//         if (type == 2) {
//             array_1d<double, 3 > /*local_angular_velocity, */local_angular_acceleration, local_torque, angular_acceleration;
// 
//             //Angular velocity and torques are saved in the local framework:
//             GeometryFunctions::QuaternionVectorGlobal2Local(Orientation, torque, local_torque);
//             GeometryFunctions::QuaternionVectorGlobal2Local(Orientation, angular_velocity, local_angular_velocity);
//             CalculateLocalAngularAccelerationByEulerEquations(i,local_angular_velocity,moments_of_inertia,local_torque, moment_reduction_factor,local_angular_acceleration);                        
// 
//             //Angular acceleration is saved in the Global framework:
//             GeometryFunctions::QuaternionVectorLocal2Global(Orientation, local_angular_acceleration, angular_acceleration);
//                     
//             UpdateRotationalVariables(StepFlag, i, rotated_angle, delta_rotation, angular_velocity, angular_acceleration, delta_t, Fix_Ang_vel);
// 
//             double ang = DEM_MODULUS_3(delta_rotation);
//                 
//             if (ang) {
//                 GeometryFunctions::UpdateOrientation(Orientation, delta_rotation);
//             } //if ang
//             GeometryFunctions::QuaternionVectorGlobal2Local(Orientation, angular_velocity, local_angular_velocity);
//         } //if type == 2
    }
           
    void DEMIntegrationScheme::UpdateTranslationalVariables(
                            int StepFlag,
                            Node < 3 > & i,
                            array_1d<double, 3 >& coor,
                            array_1d<double, 3 >& displ,
                            array_1d<double, 3 >& delta_displ,
                            array_1d<double, 3 >& vel,
                            const array_1d<double, 3 >& initial_coor,
                            const array_1d<double, 3 >& force,
                            const double force_reduction_factor,
                            const double mass,
                            const double delta_t,
                            const bool Fix_vel[3])
    {
        KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::UpdateTranslationalVariables) shouldn't be accessed, use derived class instead", 0);
    }
    
    void DEMIntegrationScheme::CalculateNewRotationalVariables(
                int StepFlag,
                Node < 3 >& i,
                const double moment_of_inertia,
                array_1d<double, 3 >& angular_velocity,
                array_1d<double, 3 >& torque,
                const double moment_reduction_factor,
                array_1d<double, 3 >& rotated_angle,
                array_1d<double, 3 >& delta_rotation,
                const double delta_t,
                const bool Fix_Ang_vel[3]) {
        KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::CalculateNewRotationalVariables) shouldn't be accessed, use derived class instead", 0);            
    }
    
    void DEMIntegrationScheme::CalculateNewRotationalVariables(
                int StepFlag,
                Node < 3 >& i,
                const array_1d<double, 3 > moments_of_inertia,
                array_1d<double, 3 >& angular_velocity,
                array_1d<double, 3 >& torque,
                const double moment_reduction_factor,
                array_1d<double, 3 >& rotated_angle,
                array_1d<double, 3 >& delta_rotation,
                Quaternion<double  >& Orientation,
                const double delta_t,
                const bool Fix_Ang_vel[3]) {
        KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::CalculateNewRotationalVariables) shouldn't be accessed, use derived class instead", 0);            
    }
    
    void DEMIntegrationScheme::UpdateLocalAngularVelocity(
                Node < 3 >& i,
                array_1d<double, 3 >& partial_local_angular_velocity,
                array_1d<double, 3 >& local_angular_velocity,
                array_1d<double, 3 >& local_angular_acceleration,
                double dt,
                const bool Fix_Ang_vel[3]) {
        KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::UpdateLocalAngularVelocity) shouldn't be accessed, use derived class instead", 0);            
    }
    
    void DEMIntegrationScheme::CalculateLocalAngularAcceleration(
                                Node < 3 >& i,
                                const double moment_of_inertia,
                                const array_1d<double, 3 >& torque, 
                                const double moment_reduction_factor,
                                array_1d<double, 3 >& angular_acceleration){
        KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::CalculateLocalAngularAcceleration) shouldn't be accessed, use derived class instead", 0);            
    }
    
    void DEMIntegrationScheme::UpdateRotationalVariables(
                int StepFlag,
                Node < 3 >& i,
                array_1d<double, 3 >& rotated_angle,
                array_1d<double, 3 >& delta_rotation,
                array_1d<double, 3 >& angular_velocity,
                array_1d<double, 3 >& angular_acceleration,
                const double delta_t,
                const bool Fix_Ang_vel[3]) {
        KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::UpdateRotationalVariables) shouldn't be accessed, use derived class instead", 0);
    }

    void DEMIntegrationScheme::UpdateRotationalVariablesOfCluster(
                Node < 3 >& i,
                const array_1d<double, 3 >& moments_of_inertia,
                array_1d<double, 3 >& rotated_angle,
                array_1d<double, 3 >& delta_rotation,
                Quaternion<double  >& Orientation,
                const array_1d<double, 3 >& angular_momentum,
                array_1d<double, 3 >& angular_velocity,
                const double delta_t,
                const bool Fix_Ang_vel[3]) {
        KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::UpdateRotationalVariablesOfCluster) shouldn't be accessed, use derived class instead", 0);
    }
    
    void DEMIntegrationScheme::UpdateRotationalVariables(
                Node < 3 >& i,
                array_1d<double, 3 >& rotated_angle,
                array_1d<double, 3 >& delta_rotation,
                const array_1d<double, 3 >& angular_velocity,
                const double delta_t,
                const bool Fix_Ang_vel[3]) {
        KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::UpdateRotationalVariables) shouldn't be accessed, use derived class instead", 0);
    }
    
    void DEMIntegrationScheme::QuaternionCalculateMidAngularVelocities(
                const Quaternion<double>& Orientation,
                const double LocalTensorInv[3][3],
                const array_1d<double, 3>& angular_momentum,
                const double dt,
                const array_1d<double, 3>& InitialAngularVel,
                array_1d<double, 3>& FinalAngularVel) {
        KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::QuaternionCalculateMidAngularVelocities) shouldn't be accessed, use derived class instead", 0);
    }
    
    void DEMIntegrationScheme::UpdateAngularVelocity(
                const Quaternion<double>& Orientation,
                const double LocalTensorInv[3][3],
                const array_1d<double, 3>& angular_momentum,
                array_1d<double, 3>& angular_velocity) {
        KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::UpdateAngularVelocity) shouldn't be accessed, use derived class instead", 0);
    }    
    
    void DEMIntegrationScheme::CalculateLocalAngularAccelerationByEulerEquations(
                                    Node < 3 >& i,
                                    const array_1d<double, 3 >& local_angular_velocity,
                                    const array_1d<double, 3 >& moments_of_inertia,
                                    const array_1d<double, 3 >& local_torque, 
                                    const double moment_reduction_factor,
                                    array_1d<double, 3 >& local_angular_acceleration){
            KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::CalculateLocalAngularAccelerationByEulerEquations) shouldn't be accessed, use derived class instead", 0);                        
    }  

    void DEMIntegrationScheme::CalculateAngularVelocityRK(
                                    const Quaternion<double  >& Orientation,
                                    const array_1d<double, 3 >& moments_of_inertia,
                                    const array_1d<double, 3 >& angular_momentum,
                                    array_1d<double, 3 >& angular_velocity,
                                    const double delta_t,
                                    const bool Fix_Ang_vel[3]) {
            KRATOS_THROW_ERROR(std::runtime_error, "This function (DEMIntegrationScheme::CalculateAngularVelocityRK) shouldn't be accessed, use derived class instead", 0);                        
    }
}

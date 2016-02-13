//
// Author: Miquel Santasusana msantasusana@cimne.upc.edu
//



#if !defined(KRATOS_DISCRETE_PARTICLE__CONFIGURE_INCLUDED)
#define  KRATOS_DISCRETE_PARTICLE__CONFIGURE_INCLUDED

// System includes
#include <string>
#include <iostream> 
#include <cmath>

// Kratos includes
#include "spatial_containers/spatial_search.h"
#include "includes/dem_variables.h"
#include "DEM_application_variables.h"

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

    
template <std::size_t TDimension>
class DiscreteParticleConfigure
{

public:
  
    enum { 
        Dimension = TDimension,
        DIMENSION = TDimension,
        MAX_LEVEL = 16,
        MIN_LEVEL = 2
    };

    /// Pointer definition of SpatialContainersConfigure
    KRATOS_CLASS_POINTER_DEFINITION(DiscreteParticleConfigure);
    
    typedef SpatialSearch                                           SearchType;

    typedef SearchType::PointType                                   PointType;
    typedef SearchType::ElementsContainerType::ContainerType        ContainerType;
    typedef SearchType::ElementsContainerType                       ElementsContainerType;
    
    typedef SearchType::ElementType                                 ElementType;
    typedef ContainerType::value_type                               PointerType;
    typedef ContainerType::iterator                                 IteratorType;
    typedef ElementsContainerType::iterator                         ElementIteratorType;
    
    typedef SearchType::ElementsContainerType::ContainerType        ResultContainerType;
//     typedef SearchType::ResultDistanceType::ContainerType             ResultDistanceType;
    
    typedef ResultContainerType::iterator                           ResultIteratorType;
    typedef std::vector<double>::iterator                           DistanceIteratorType;
    
    typedef ContactPair<PointerType>                                ContactPairType;
    typedef std::vector<ContactPairType>                            ContainerContactType;
    typedef ContainerContactType::iterator                          IteratorContactType;
    typedef ContainerContactType::value_type                        PointerContactType;
    
    /////////////////////////////////////////////////////////////////////////////////////////// bins_dynamic_objects.h:279
    
//     typedef SpatialSearch                                                       SearchType;
// 
//     typedef SearchType::PointType                                               PointType;
//     typedef PointerVectorSet<GeometricalObject, IndexedObject>::ContainerType   ContainerType;
//     typedef PointerVectorSet<GeometricalObject, IndexedObject>                  ElementsContainerType;
//     
//     typedef SearchType::ElementType                                             ElementType;
//     typedef ContainerType::value_type                                           PointerType;
//     typedef ContainerType::iterator                                             IteratorType;
//     typedef ElementsContainerType::iterator                                     ElementIteratorType;
//     
//     typedef PointerVectorSet<GeometricalObject, IndexedObject>::ContainerType   ResultContainerType;
// //     typedef SearchType::ResultDistanceType::ContainerType             ResultDistanceType;
//     
//     typedef ResultContainerType::iterator                           ResultIteratorType;
//     typedef std::vector<double>::iterator                           DistanceIteratorType;
//     
//     typedef ContactPair<PointerType>                                ContactPairType;
//     typedef std::vector<ContactPairType>                            ContainerContactType;
//     typedef ContainerContactType::iterator                          IteratorContactType;
//     typedef ContainerContactType::value_type                        PointerContactType;
    
    ///@}
    ///@name Life Cycle
    ///@{

    DiscreteParticleConfigure(){};
    virtual ~DiscreteParticleConfigure(){
	}

    ///@}
    ///@name Operators
    ///@{


    ///@}
    ///@name Operations
    ///@{

    //******************************************************************************************************************

    static inline void CalculateBoundingBox(const PointerType& rObject, PointType& rLowPoint, PointType& rHighPoint)
    {    
        rHighPoint = rLowPoint  = rObject->GetGeometry()[0];
        double radius = rObject->GetGeometry()[0].FastGetSolutionStepValue(RADIUS);

        for(std::size_t i = 0; i < 3; i++)
        {
            rLowPoint[i]  += -radius;
            rHighPoint[i] += radius;
        }
    }

    static inline void CalculateBoundingBox(const PointerType& rObject, PointType& rLowPoint, PointType& rHighPoint, const double& Radius)
    {
        rHighPoint = rLowPoint  = rObject->GetGeometry()[0];
        
        for(std::size_t i = 0; i < 3; i++)
        {
            rLowPoint[i]  += -Radius;
            rHighPoint[i] += Radius;
        }
    }
        
    static inline void CalculateCenter(const PointerType& rObject, PointType& rCenter)
    {
        rCenter  = rObject->GetGeometry()[0];
    }

    //******************************************************************************************************************

    static inline bool Intersection(const PointerType& rObj_1, const PointerType& rObj_2)
    {
        array_1d<double, 3> rObj_2_to_rObj_1 = rObj_1->GetGeometry()[0] - rObj_2->GetGeometry()[0];
        double distance_2 = inner_prod(rObj_2_to_rObj_1, rObj_2_to_rObj_1);

        const double& radius_1 = rObj_1->GetGeometry()[0].FastGetSolutionStepValue(RADIUS);
        const double& radius_2 = rObj_2->GetGeometry()[0].FastGetSolutionStepValue(RADIUS);
        double radius_sum      = radius_1 + radius_2;
        bool intersect         = floatle((distance_2 - radius_sum * radius_sum),0);
        return intersect;
    }

    static inline bool Intersection(const PointerType& rObj_1, const PointerType& rObj_2, const double& radius_1)
    {
        //array_1d<double, 3> rObj_2_to_rObj_1 = rObj_1->GetGeometry()[0] - rObj_2->GetGeometry()[0];
        const array_1d<double, 3> coor1 = rObj_1->GetGeometry()[0];
        const array_1d<double, 3> coor2 = rObj_2->GetGeometry()[0];

        double difference[3];
        difference[0] = coor1[0] - coor2[0];
        difference[1] = coor1[1] - coor2[1];
        difference[2] = coor1[2] - coor2[2];
        double distance_2 = DEM_INNER_PRODUCT_3(difference, difference);

        double radius_sum      = radius_1 + rObj_2->GetGeometry()[0].FastGetSolutionStepValue(RADIUS);
        bool intersect         = floatle((distance_2 - radius_sum * radius_sum),0);

        return intersect;
    }

    //******************************************************************************************************************
    
    static inline bool  IntersectionBox(const PointerType& rObject,  const PointType& rLowPoint, const PointType& rHighPoint)
    {
 
//        double separation_from_particle_radius_ratio = 0.1;

        array_1d<double, 3> center_of_particle = rObject->GetGeometry()[0];
 
        const double& radius = rObject->GetGeometry()[0].FastGetSolutionStepValue(RADIUS);

        bool intersect = (
          floatle(rLowPoint[0]  - radius,center_of_particle[0]) && 
          floatle(rLowPoint[1]  - radius,center_of_particle[1]) && 
          floatle(rLowPoint[2]  - radius,center_of_particle[2]) &&
          floatge(rHighPoint[0] + radius,center_of_particle[0]) && 
          floatge(rHighPoint[1] + radius,center_of_particle[1]) && 
          floatge(rHighPoint[2] + radius,center_of_particle[2]));

        return  intersect;
 
    }

    static inline bool  IntersectionBox(const PointerType& rObject,  const PointType& rLowPoint, const PointType& rHighPoint, const double& Radius)
    {
//        double separation_from_particle_radius_ratio = 0.1;
        array_1d<double, 3> center_of_particle = rObject->GetGeometry()[0];

        double radius = Radius;//Cambien el radi del objecte de cerca per el gran, aixi no tindria que petar res
        bool intersect = (
          floatle(rLowPoint[0]  - radius,center_of_particle[0]) && 
          floatle(rLowPoint[1]  - radius,center_of_particle[1]) && 
          floatle(rLowPoint[2]  - radius,center_of_particle[2]) &&
          floatge(rHighPoint[0] + radius,center_of_particle[0]) && 
          floatge(rHighPoint[1] + radius,center_of_particle[1]) && 
          floatge(rHighPoint[2] + radius,center_of_particle[2]));
        return  intersect;
    }

    static inline void Distance(const PointerType& rObj_1, const PointerType& rObj_2, double& distance)
    {
        array_1d<double, 3> center_of_particle1 = rObj_1->GetGeometry()[0];
        array_1d<double, 3> center_of_particle2 = rObj_2->GetGeometry()[0];

        distance = sqrt((center_of_particle1[0] - center_of_particle2[0]) * (center_of_particle1[0] - center_of_particle2[0]) +
                        (center_of_particle1[1] - center_of_particle2[1]) * (center_of_particle1[1] - center_of_particle2[1]) +
                        (center_of_particle1[2] - center_of_particle2[2]) * (center_of_particle1[2] - center_of_particle2[2]) );
    }
     
    //******************************************************************************************************************

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
    virtual std::string Info() const {return " Spatial Containers Configure for Particles"; }

    /// Print information about this object.
    virtual void PrintInfo(std::ostream& rOStream) const {}

    /// Print object's data.
    virtual void PrintData(std::ostream& rOStream) const {}

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
    ///@name Private Operators
    ///@{

    ///@}
    ///@name Private Operations
    ///@{
      
    static inline bool floateq(double a, double b) {
        return std::fabs(a - b) < std::numeric_limits<double>::epsilon();
    }
    
    static inline bool floatle(double a, double b) {
        return a < b || std::fabs(a - b) < std::numeric_limits<double>::epsilon();
    }
    
    static inline bool floatge(double a, double b) {
        return a > b || std::fabs(a - b) < std::numeric_limits<double>::epsilon();
    }

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
    DiscreteParticleConfigure& operator=(DiscreteParticleConfigure const& rOther);

    /// Copy constructor.
    DiscreteParticleConfigure(DiscreteParticleConfigure const& rOther);

    ///@}

    }; // Class ParticleConfigure

    ///@}

    ///@name Type Definitions
    ///@{

    ///@}
    ///@name Input and output
    ///@{

    /// input stream function
    template <std::size_t TDimension>
    inline std::istream& operator >> (std::istream& rIStream, DiscreteParticleConfigure<TDimension> & rThis){
        return rIStream;
        }

    /// output stream function
    template <std::size_t TDimension>
    inline std::ostream& operator << (std::ostream& rOStream, const DiscreteParticleConfigure<TDimension>& rThis){
        rThis.PrintInfo(rOStream);
        rOStream << std::endl;
        rThis.PrintData(rOStream);

        return rOStream;
        }

    ///@}

}   // namespace Kratos.
#endif	/* DISCRETE_PARTICLE_CONFIGURE_H */

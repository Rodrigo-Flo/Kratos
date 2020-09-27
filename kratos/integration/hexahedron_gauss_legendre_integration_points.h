//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//
//


#if !defined(KRATOS_HEXAHEDRON_GAUSS_LEGENDRE_INTEGRATION_POINTS_H_INCLUDED )
#define  KRATOS_HEXAHEDRON_GAUSS_LEGENDRE_INTEGRATION_POINTS_H_INCLUDED

// System includes

// External includes

// Project includes
#include "integration/quadrature.h"


namespace Kratos
{

class KRATOS_API(KRATOS_CORE) HexahedronGaussLegendreIntegrationPoints1
{
public:
    KRATOS_CLASS_POINTER_DEFINITION(HexahedronGaussLegendreIntegrationPoints1);
    typedef std::size_t SizeType;

    static const unsigned int Dimension = 3;

    typedef IntegrationPoint<3> IntegrationPointType;

    typedef std::array<IntegrationPointType, 1> IntegrationPointsArrayType;

    typedef IntegrationPointType::PointType PointType;

    static SizeType IntegrationPointsNumber()
    {
        return 1;
    }

    static const IntegrationPointsArrayType GenerateIntegrationPoints()
    {
        static const IntegrationPointsArrayType s_integration_points{{
            IntegrationPointType( 0.00 , 0.00, 0.00 , 8.00 )
        }};
        return s_integration_points;
    }

    static IntegrationPointsArrayType& IntegrationPoints()
    {
        return msIntegrationPoints;
    }

    std::string Info() const
    {
        std::stringstream buffer;
        buffer << "Hexahedron Gauss-Legendre quadrature 1 ";
        return buffer.str();
    }
protected:

private:

    static IntegrationPointsArrayType msIntegrationPoints;

}; // Class HexahedronGaussLegendreIntegrationPoints1

class KRATOS_API(KRATOS_CORE) HexahedronGaussLegendreIntegrationPoints2
{
public:
    KRATOS_CLASS_POINTER_DEFINITION(HexahedronGaussLegendreIntegrationPoints2);
    typedef std::size_t SizeType;

    static const unsigned int Dimension = 3;

    typedef IntegrationPoint<3> IntegrationPointType;

    typedef std::array<IntegrationPointType, 8> IntegrationPointsArrayType;

    typedef IntegrationPointType::PointType PointType;

    static SizeType IntegrationPointsNumber()
    {
        return 8;
    }

    static const IntegrationPointsArrayType GenerateIntegrationPoints()
    {
        static const IntegrationPointsArrayType s_integration_points{{
            IntegrationPointType( -1.00/std::sqrt(3.0) , -1.00/std::sqrt(3.0), -1.00/std::sqrt(3.0), 1.00 ),
            IntegrationPointType(  1.00/std::sqrt(3.0) , -1.00/std::sqrt(3.0), -1.00/std::sqrt(3.0), 1.00 ),
            IntegrationPointType(  1.00/std::sqrt(3.0) ,  1.00/std::sqrt(3.0), -1.00/std::sqrt(3.0), 1.00 ),
            IntegrationPointType( -1.00/std::sqrt(3.0) ,  1.00/std::sqrt(3.0), -1.00/std::sqrt(3.0), 1.00 ),
            IntegrationPointType( -1.00/std::sqrt(3.0) , -1.00/std::sqrt(3.0),  1.00/std::sqrt(3.0), 1.00 ),
            IntegrationPointType(  1.00/std::sqrt(3.0) , -1.00/std::sqrt(3.0),  1.00/std::sqrt(3.0), 1.00 ),
            IntegrationPointType(  1.00/std::sqrt(3.0) ,  1.00/std::sqrt(3.0),  1.00/std::sqrt(3.0), 1.00 ),
            IntegrationPointType( -1.00/std::sqrt(3.0) ,  1.00/std::sqrt(3.0),  1.00/std::sqrt(3.0), 1.00 )
        }};
        return s_integration_points;
    }

    static IntegrationPointsArrayType& IntegrationPoints()
    {
        return msIntegrationPoints;
    }

    std::string Info() const
    {
        std::stringstream buffer;
        buffer << "Hexahedron Gauss-Legendre quadrature 2 ";
        return buffer.str();
    }
protected:

private:

    static IntegrationPointsArrayType msIntegrationPoints;

}; // Class HexahedronGaussLegendreIntegrationPoints2

class KRATOS_API(KRATOS_CORE) HexahedronGaussLegendreIntegrationPoints3
{
public:
    KRATOS_CLASS_POINTER_DEFINITION(HexahedronGaussLegendreIntegrationPoints3);
    typedef std::size_t SizeType;

    static const unsigned int Dimension = 3;

    typedef IntegrationPoint<3> IntegrationPointType;

    typedef std::array<IntegrationPointType, 27> IntegrationPointsArrayType;

    typedef IntegrationPointType::PointType PointType;

    static SizeType IntegrationPointsNumber()
    {
        return 27;
    }

    static const IntegrationPointsArrayType GenerateIntegrationPoints()
    {
        static const IntegrationPointsArrayType s_integration_points{{
            IntegrationPointType( -std::sqrt(3.00/5.00) , -std::sqrt(3.00/5.00), -std::sqrt(3.00/5.00), 125.00/729.00 ),
            IntegrationPointType(                   0.0 , -std::sqrt(3.00/5.00), -std::sqrt(3.00/5.00), 200.00/729.00 ),
            IntegrationPointType(  std::sqrt(3.00/5.00) , -std::sqrt(3.00/5.00), -std::sqrt(3.00/5.00), 125.00/729.00 ),

            IntegrationPointType( -std::sqrt(3.00/5.00) ,                   0.0, -std::sqrt(3.00/5.00), 200.00/729.00 ),
            IntegrationPointType(                   0.0 ,                   0.0, -std::sqrt(3.00/5.00), 320.00/729.00 ),
            IntegrationPointType(  std::sqrt(3.00/5.00) ,                   0.0, -std::sqrt(3.00/5.00), 200.00/729.00 ),

            IntegrationPointType( -std::sqrt(3.00/5.00) ,  std::sqrt(3.00/5.00), -std::sqrt(3.00/5.00), 125.00/729.00 ),
            IntegrationPointType(                   0.0 ,  std::sqrt(3.00/5.00), -std::sqrt(3.00/5.00), 200.00/729.00 ),
            IntegrationPointType(  std::sqrt(3.00/5.00) ,  std::sqrt(3.00/5.00), -std::sqrt(3.00/5.00), 125.00/729.00 ),

            IntegrationPointType( -std::sqrt(3.00/5.00) , -std::sqrt(3.00/5.00),                   0.0, 200.00/729.00 ),
            IntegrationPointType(                   0.0 , -std::sqrt(3.00/5.00),                   0.0, 320.00/729.00 ),
            IntegrationPointType(  std::sqrt(3.00/5.00) , -std::sqrt(3.00/5.00),                   0.0, 200.00/729.00 ),

            IntegrationPointType( -std::sqrt(3.00/5.00) ,                   0.0,                   0.0, 320.00/729.00 ),
            IntegrationPointType(                   0.0 ,                   0.0,                   0.0, 512.00/729.00 ),
            IntegrationPointType(  std::sqrt(3.00/5.00) ,                   0.0,                   0.0, 320.00/729.00 ),

            IntegrationPointType( -std::sqrt(3.00/5.00) ,  std::sqrt(3.00/5.00),                   0.0, 200.00/729.00 ),
            IntegrationPointType(                   0.0 ,  std::sqrt(3.00/5.00),                   0.0, 320.00/729.00 ),
            IntegrationPointType(  std::sqrt(3.00/5.00) ,  std::sqrt(3.00/5.00),                   0.0, 200.00/729.00 ),

            IntegrationPointType( -std::sqrt(3.00/5.00) , -std::sqrt(3.00/5.00),  std::sqrt(3.00/5.00), 125.00/729.00 ),
            IntegrationPointType(                   0.0 , -std::sqrt(3.00/5.00),  std::sqrt(3.00/5.00), 200.00/729.00 ),
            IntegrationPointType(  std::sqrt(3.00/5.00) , -std::sqrt(3.00/5.00),  std::sqrt(3.00/5.00), 125.00/729.00 ),

            IntegrationPointType( -std::sqrt(3.00/5.00) ,                   0.0,  std::sqrt(3.00/5.00), 200.00/729.00 ),
            IntegrationPointType(                   0.0 ,                   0.0,  std::sqrt(3.00/5.00), 320.00/729.00 ),
            IntegrationPointType(  std::sqrt(3.00/5.00) ,                   0.0,  std::sqrt(3.00/5.00), 200.00/729.00 ),

            IntegrationPointType( -std::sqrt(3.00/5.00) ,  std::sqrt(3.00/5.00),  std::sqrt(3.00/5.00), 125.00/729.00 ),
            IntegrationPointType(                   0.0 ,  std::sqrt(3.00/5.00),  std::sqrt(3.00/5.00), 200.00/729.00 ),
            IntegrationPointType(  std::sqrt(3.00/5.00) ,  std::sqrt(3.00/5.00),  std::sqrt(3.00/5.00), 125.00/729.00 )
        }};
        return s_integration_points;
    }

    static IntegrationPointsArrayType& IntegrationPoints()
    {
        return msIntegrationPoints;
    }

    std::string Info() const
    {
        std::stringstream buffer;
        buffer << "Hexadra Gauss-Legendre quadrature 3 ";
        return buffer.str();
    }
protected:

private:

    static IntegrationPointsArrayType msIntegrationPoints;

}; // Class HexahedronGaussLegendreIntegrationPoints3

class KRATOS_API(KRATOS_CORE) HexahedronGaussLegendreIntegrationPoints4
{
public:
    KRATOS_CLASS_POINTER_DEFINITION(HexahedronGaussLegendreIntegrationPoints4);
    typedef std::size_t SizeType;

    static const unsigned int Dimension = 3;

    typedef IntegrationPoint<3> IntegrationPointType;

    typedef std::array<IntegrationPointType, 64> IntegrationPointsArrayType;

    typedef IntegrationPointType::PointType PointType;

    static SizeType IntegrationPointsNumber()
    {
        return 64;
    }

    static const IntegrationPointsArrayType GenerateIntegrationPoints()
    {
        static const IntegrationPointsArrayType s_integration_points{{
            IntegrationPointType(-0.86113631159405257522 , -0.86113631159405257522 , -0.86113631159405257522 , 0.04209147749053145454),
            IntegrationPointType(-0.33998104358485626480 , -0.86113631159405257522 , -0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(0.33998104358485626480 , -0.86113631159405257522 , -0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(0.86113631159405257522 , -0.86113631159405257522 , -0.86113631159405257522 , 0.04209147749053145454),
            IntegrationPointType(-0.86113631159405257522 , -0.33998104358485626480 , -0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(-0.33998104358485626480 , -0.33998104358485626480 , -0.86113631159405257522 , 0.14794033605678130087),
            IntegrationPointType(0.33998104358485626480 , -0.33998104358485626480 , -0.86113631159405257522 , 0.14794033605678130087),
            IntegrationPointType(0.86113631159405257522 , -0.33998104358485626480 , -0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(-0.86113631159405257522 , 0.33998104358485626480 , -0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(-0.33998104358485626480 , 0.33998104358485626480 , -0.86113631159405257522 , 0.14794033605678130087),
            IntegrationPointType(0.33998104358485626480 , 0.33998104358485626480 , -0.86113631159405257522 , 0.14794033605678130087),
            IntegrationPointType(0.86113631159405257522 , 0.33998104358485626480 , -0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(-0.86113631159405257522 , 0.86113631159405257522 , -0.86113631159405257522 , 0.04209147749053145454),
            IntegrationPointType(-0.33998104358485626480 , 0.86113631159405257522 , -0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(0.33998104358485626480 , 0.86113631159405257522 , -0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(0.86113631159405257522 , 0.86113631159405257522 , -0.86113631159405257522 , 0.04209147749053145454),
            IntegrationPointType(-0.86113631159405257522 , -0.86113631159405257522 , -0.33998104358485626480 , 0.07891151579507055098),
            IntegrationPointType(-0.33998104358485626480 , -0.86113631159405257522 , -0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(0.33998104358485626480 , -0.86113631159405257522 , -0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(0.86113631159405257522 , -0.86113631159405257522 , -0.33998104358485626480 , 0.07891151579507055098),
            IntegrationPointType(-0.86113631159405257522 , -0.33998104358485626480 , -0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(-0.33998104358485626480 , -0.33998104358485626480 , -0.33998104358485626480 , 0.27735296695391298990),
            IntegrationPointType(0.33998104358485626480 , -0.33998104358485626480 , -0.33998104358485626480 , 0.27735296695391298990),
            IntegrationPointType(0.86113631159405257522 , -0.33998104358485626480 , -0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(-0.86113631159405257522 , 0.33998104358485626480 , -0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(-0.33998104358485626480 , 0.33998104358485626480 , -0.33998104358485626480 , 0.27735296695391298990),
            IntegrationPointType(0.33998104358485626480 , 0.33998104358485626480 , -0.33998104358485626480 , 0.27735296695391298990),
            IntegrationPointType(0.86113631159405257522 , 0.33998104358485626480 , -0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(-0.86113631159405257522 , 0.86113631159405257522 , -0.33998104358485626480 , 0.07891151579507055098),
            IntegrationPointType(-0.33998104358485626480 , 0.86113631159405257522 , -0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(0.33998104358485626480 , 0.86113631159405257522 , -0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(0.86113631159405257522 , 0.86113631159405257522 , -0.33998104358485626480 , 0.07891151579507055098),
            IntegrationPointType(-0.86113631159405257522 , -0.86113631159405257522 , 0.33998104358485626480 , 0.07891151579507055098),
            IntegrationPointType(-0.33998104358485626480 , -0.86113631159405257522 , 0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(0.33998104358485626480 , -0.86113631159405257522 , 0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(0.86113631159405257522 , -0.86113631159405257522 , 0.33998104358485626480 , 0.07891151579507055098),
            IntegrationPointType(-0.86113631159405257522 , -0.33998104358485626480 , 0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(-0.33998104358485626480 , -0.33998104358485626480 , 0.33998104358485626480 , 0.27735296695391298990),
            IntegrationPointType(0.33998104358485626480 , -0.33998104358485626480 , 0.33998104358485626480 , 0.27735296695391298990),
            IntegrationPointType(0.86113631159405257522 , -0.33998104358485626480 , 0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(-0.86113631159405257522 , 0.33998104358485626480 , 0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(-0.33998104358485626480 , 0.33998104358485626480 , 0.33998104358485626480 , 0.27735296695391298990),
            IntegrationPointType(0.33998104358485626480 , 0.33998104358485626480 , 0.33998104358485626480 , 0.27735296695391298990),
            IntegrationPointType(0.86113631159405257522 , 0.33998104358485626480 , 0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(-0.86113631159405257522 , 0.86113631159405257522 , 0.33998104358485626480 , 0.07891151579507055098),
            IntegrationPointType(-0.33998104358485626480 , 0.86113631159405257522 , 0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(0.33998104358485626480 , 0.86113631159405257522 , 0.33998104358485626480 , 0.14794033605678130087),
            IntegrationPointType(0.86113631159405257522 , 0.86113631159405257522 , 0.33998104358485626480 , 0.07891151579507055098),
            IntegrationPointType(-0.86113631159405257522 , -0.86113631159405257522 , 0.86113631159405257522 , 0.04209147749053145454),
            IntegrationPointType(-0.33998104358485626480 , -0.86113631159405257522 , 0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(0.33998104358485626480 , -0.86113631159405257522 , 0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(0.86113631159405257522 , -0.86113631159405257522 , 0.86113631159405257522 , 0.04209147749053145454),
            IntegrationPointType(-0.86113631159405257522 , -0.33998104358485626480 , 0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(-0.33998104358485626480 , -0.33998104358485626480 , 0.86113631159405257522 , 0.14794033605678130087),
            IntegrationPointType(0.33998104358485626480 , -0.33998104358485626480 , 0.86113631159405257522 , 0.14794033605678130087),
            IntegrationPointType(0.86113631159405257522 , -0.33998104358485626480 , 0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(-0.86113631159405257522 , 0.33998104358485626480 , 0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(-0.33998104358485626480 , 0.33998104358485626480 , 0.86113631159405257522 , 0.14794033605678130087),
            IntegrationPointType(0.33998104358485626480 , 0.33998104358485626480 , 0.86113631159405257522 , 0.14794033605678130087),
            IntegrationPointType(0.86113631159405257522 , 0.33998104358485626480 , 0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(-0.86113631159405257522 , 0.86113631159405257522 , 0.86113631159405257522 , 0.04209147749053145454),
            IntegrationPointType(-0.33998104358485626480 , 0.86113631159405257522 , 0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(0.33998104358485626480 , 0.86113631159405257522 , 0.86113631159405257522 , 0.07891151579507055098),
            IntegrationPointType(0.86113631159405257522 , 0.86113631159405257522 , 0.86113631159405257522 , 0.04209147749053145454)
        }};
        return s_integration_points;
    }

    static IntegrationPointsArrayType& IntegrationPoints()
    {
        return msIntegrationPoints;
    }

    std::string Info() const
    {
        std::stringstream buffer;
        buffer << "Hexadra Gauss-Legendre quadrature 4 ";
        return buffer.str();
    }
protected:

private:

   static IntegrationPointsArrayType msIntegrationPoints;

}; // Class HexahedronGaussLegendreIntegrationPoints4

class KRATOS_API(KRATOS_CORE) HexahedronGaussLegendreIntegrationPoints5
{
public:
   KRATOS_CLASS_POINTER_DEFINITION(HexahedronGaussLegendreIntegrationPoints5);
   typedef std::size_t SizeType;

   static const unsigned int Dimension = 3;

   typedef IntegrationPoint<3> IntegrationPointType;

   typedef std::array<IntegrationPointType, 125> IntegrationPointsArrayType;

   typedef IntegrationPointType::PointType PointType;

   static SizeType IntegrationPointsNumber()
   {
       return 125;
   }

    static const IntegrationPointsArrayType GenerateIntegrationPoints()
    {
        static const IntegrationPointsArrayType s_integration_points{{
            IntegrationPointType(-0.90617984593866399280 , -0.90617984593866399280 , -0.90617984593866399280 , 0.013299736420632648092),
            IntegrationPointType(-0.53846931010568309104 , -0.90617984593866399280 , -0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(0 , -0.90617984593866399280 , -0.90617984593866399280 , 0.031934207352848290676),
            IntegrationPointType(0.53846931010568309104 , -0.90617984593866399280 , -0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(0.90617984593866399280 , -0.90617984593866399280 , -0.90617984593866399280 , 0.013299736420632648092),
            IntegrationPointType(-0.90617984593866399280 , -0.53846931010568309104 , -0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(-0.53846931010568309104 , -0.53846931010568309104 , -0.90617984593866399280 , 0.05427649123462815748),
            IntegrationPointType(0 , -0.53846931010568309104 , -0.90617984593866399280 , 0.06451200000000000000),
            IntegrationPointType(0.53846931010568309104 , -0.53846931010568309104 , -0.90617984593866399280 , 0.05427649123462815748),
            IntegrationPointType(0.90617984593866399280 , -0.53846931010568309104 , -0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(-0.90617984593866399280 , 0 , -0.90617984593866399280 , 0.031934207352848290676),
            IntegrationPointType(-0.53846931010568309104 , 0 , -0.90617984593866399280 , 0.06451200000000000000),
            IntegrationPointType(0 , 0 , -0.90617984593866399280 , 0.07667773006934522489),
            IntegrationPointType(0.53846931010568309104 , 0 , -0.90617984593866399280 , 0.06451200000000000000),
            IntegrationPointType(0.90617984593866399280 , 0 , -0.90617984593866399280 , 0.031934207352848290676),
            IntegrationPointType(-0.90617984593866399280 , 0.53846931010568309104 , -0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(-0.53846931010568309104 , 0.53846931010568309104 , -0.90617984593866399280 , 0.05427649123462815748),
            IntegrationPointType(0 , 0.53846931010568309104 , -0.90617984593866399280 , 0.06451200000000000000),
            IntegrationPointType(0.53846931010568309104 , 0.53846931010568309104 , -0.90617984593866399280 , 0.05427649123462815748),
            IntegrationPointType(0.90617984593866399280 , 0.53846931010568309104 , -0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(-0.90617984593866399280 , 0.90617984593866399280 , -0.90617984593866399280 , 0.013299736420632648092),
            IntegrationPointType(-0.53846931010568309104 , 0.90617984593866399280 , -0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(0 , 0.90617984593866399280 , -0.90617984593866399280 , 0.031934207352848290676),
            IntegrationPointType(0.53846931010568309104 , 0.90617984593866399280 , -0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(0.90617984593866399280 , 0.90617984593866399280 , -0.90617984593866399280 , 0.013299736420632648092),
            IntegrationPointType(-0.90617984593866399280 , -0.90617984593866399280 , -0.53846931010568309104 , 0.026867508765371842524),
            IntegrationPointType(-0.53846931010568309104 , -0.90617984593866399280 , -0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(0 , -0.90617984593866399280 , -0.53846931010568309104 , 0.06451200000000000000),
            IntegrationPointType(0.53846931010568309104 , -0.90617984593866399280 , -0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(0.90617984593866399280 , -0.90617984593866399280 , -0.53846931010568309104 , 0.026867508765371842524),
            IntegrationPointType(-0.90617984593866399280 , -0.53846931010568309104 , -0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(-0.53846931010568309104 , -0.53846931010568309104 , -0.53846931010568309104 , 0.10964684245453881967),
            IntegrationPointType(0 , -0.53846931010568309104 , -0.53846931010568309104 , 0.13032414106964827997),
            IntegrationPointType(0.53846931010568309104 , -0.53846931010568309104 , -0.53846931010568309104 , 0.10964684245453881967),
            IntegrationPointType(0.90617984593866399280 , -0.53846931010568309104 , -0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(-0.90617984593866399280 , 0 , -0.53846931010568309104 , 0.06451200000000000000),
            IntegrationPointType(-0.53846931010568309104 , 0 , -0.53846931010568309104 , 0.13032414106964827997),
            IntegrationPointType(0 , 0 , -0.53846931010568309104 , 0.15490078296220484370),
            IntegrationPointType(0.53846931010568309104 , 0 , -0.53846931010568309104 , 0.13032414106964827997),
            IntegrationPointType(0.90617984593866399280 , 0 , -0.53846931010568309104 , 0.06451200000000000000),
            IntegrationPointType(-0.90617984593866399280 , 0.53846931010568309104 , -0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(-0.53846931010568309104 , 0.53846931010568309104 , -0.53846931010568309104 , 0.10964684245453881967),
            IntegrationPointType(0 , 0.53846931010568309104 , -0.53846931010568309104 , 0.13032414106964827997),
            IntegrationPointType(0.53846931010568309104 , 0.53846931010568309104 , -0.53846931010568309104 , 0.10964684245453881967),
            IntegrationPointType(0.90617984593866399280 , 0.53846931010568309104 , -0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(-0.90617984593866399280 , 0.90617984593866399280 , -0.53846931010568309104 , 0.026867508765371842524),
            IntegrationPointType(-0.53846931010568309104 , 0.90617984593866399280 , -0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(0 , 0.90617984593866399280 , -0.53846931010568309104 , 0.06451200000000000000),
            IntegrationPointType(0.53846931010568309104 , 0.90617984593866399280 , -0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(0.90617984593866399280 , 0.90617984593866399280 , -0.53846931010568309104 , 0.026867508765371842524),
            IntegrationPointType(-0.90617984593866399280 , -0.90617984593866399280 , 0 , 0.031934207352848290676),
            IntegrationPointType(-0.53846931010568309104 , -0.90617984593866399280 , 0 , 0.06451200000000000000),
            IntegrationPointType(0 , -0.90617984593866399280 , 0 , 0.07667773006934522489),
            IntegrationPointType(0.53846931010568309104 , -0.90617984593866399280 , 0 , 0.06451200000000000000),
            IntegrationPointType(0.90617984593866399280 , -0.90617984593866399280 , 0 , 0.031934207352848290676),
            IntegrationPointType(-0.90617984593866399280 , -0.53846931010568309104 , 0 , 0.06451200000000000000),
            IntegrationPointType(-0.53846931010568309104 , -0.53846931010568309104 , 0 , 0.13032414106964827997),
            IntegrationPointType(0 , -0.53846931010568309104 , 0 , 0.15490078296220484370),
            IntegrationPointType(0.53846931010568309104 , -0.53846931010568309104 , 0 , 0.13032414106964827997),
            IntegrationPointType(0.90617984593866399280 , -0.53846931010568309104 , 0 , 0.06451200000000000000),
            IntegrationPointType(-0.90617984593866399280 , 0 , 0 , 0.07667773006934522489),
            IntegrationPointType(-0.53846931010568309104 , 0 , 0 , 0.15490078296220484370),
            IntegrationPointType(0 , 0 , 0 , 0.18411210973936899863),
            IntegrationPointType(0.53846931010568309104 , 0 , 0 , 0.15490078296220484370),
            IntegrationPointType(0.90617984593866399280 , 0 , 0 , 0.07667773006934522489),
            IntegrationPointType(-0.90617984593866399280 , 0.53846931010568309104 , 0 , 0.06451200000000000000),
            IntegrationPointType(-0.53846931010568309104 , 0.53846931010568309104 , 0 , 0.13032414106964827997),
            IntegrationPointType(0 , 0.53846931010568309104 , 0 , 0.15490078296220484370),
            IntegrationPointType(0.53846931010568309104 , 0.53846931010568309104 , 0 , 0.13032414106964827997),
            IntegrationPointType(0.90617984593866399280 , 0.53846931010568309104 , 0 , 0.06451200000000000000),
            IntegrationPointType(-0.90617984593866399280 , 0.90617984593866399280 , 0 , 0.031934207352848290676),
            IntegrationPointType(-0.53846931010568309104 , 0.90617984593866399280 , 0 , 0.06451200000000000000),
            IntegrationPointType(0 , 0.90617984593866399280 , 0 , 0.07667773006934522489),
            IntegrationPointType(0.53846931010568309104 , 0.90617984593866399280 , 0 , 0.06451200000000000000),
            IntegrationPointType(0.90617984593866399280 , 0.90617984593866399280 , 0 , 0.031934207352848290676),
            IntegrationPointType(-0.90617984593866399280 , -0.90617984593866399280 , 0.53846931010568309104 , 0.026867508765371842524),
            IntegrationPointType(-0.53846931010568309104 , -0.90617984593866399280 , 0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(0 , -0.90617984593866399280 , 0.53846931010568309104 , 0.06451200000000000000),
            IntegrationPointType(0.53846931010568309104 , -0.90617984593866399280 , 0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(0.90617984593866399280 , -0.90617984593866399280 , 0.53846931010568309104 , 0.026867508765371842524),
            IntegrationPointType(-0.90617984593866399280 , -0.53846931010568309104 , 0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(-0.53846931010568309104 , -0.53846931010568309104 , 0.53846931010568309104 , 0.10964684245453881967),
            IntegrationPointType(0 , -0.53846931010568309104 , 0.53846931010568309104 , 0.13032414106964827997),
            IntegrationPointType(0.53846931010568309104 , -0.53846931010568309104 , 0.53846931010568309104 , 0.10964684245453881967),
            IntegrationPointType(0.90617984593866399280 , -0.53846931010568309104 , 0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(-0.90617984593866399280 , 0 , 0.53846931010568309104 , 0.06451200000000000000),
            IntegrationPointType(-0.53846931010568309104 , 0 , 0.53846931010568309104 , 0.13032414106964827997),
            IntegrationPointType(0 , 0 , 0.53846931010568309104 , 0.15490078296220484370),
            IntegrationPointType(0.53846931010568309104 , 0 , 0.53846931010568309104 , 0.13032414106964827997),
            IntegrationPointType(0.90617984593866399280 , 0 , 0.53846931010568309104 , 0.06451200000000000000),
            IntegrationPointType(-0.90617984593866399280 , 0.53846931010568309104 , 0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(-0.53846931010568309104 , 0.53846931010568309104 , 0.53846931010568309104 , 0.10964684245453881967),
            IntegrationPointType(0 , 0.53846931010568309104 , 0.53846931010568309104 , 0.13032414106964827997),
            IntegrationPointType(0.53846931010568309104 , 0.53846931010568309104 , 0.53846931010568309104 , 0.10964684245453881967),
            IntegrationPointType(0.90617984593866399280 , 0.53846931010568309104 , 0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(-0.90617984593866399280 , 0.90617984593866399280 , 0.53846931010568309104 , 0.026867508765371842524),
            IntegrationPointType(-0.53846931010568309104 , 0.90617984593866399280 , 0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(0 , 0.90617984593866399280 , 0.53846931010568309104 , 0.06451200000000000000),
            IntegrationPointType(0.53846931010568309104 , 0.90617984593866399280 , 0.53846931010568309104 , 0.05427649123462815748),
            IntegrationPointType(0.90617984593866399280 , 0.90617984593866399280 , 0.53846931010568309104 , 0.026867508765371842524),
            IntegrationPointType(-0.90617984593866399280 , -0.90617984593866399280 , 0.90617984593866399280 , 0.013299736420632648092),
            IntegrationPointType(-0.53846931010568309104 , -0.90617984593866399280 , 0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(0 , -0.90617984593866399280 , 0.90617984593866399280 , 0.031934207352848290676),
            IntegrationPointType(0.53846931010568309104 , -0.90617984593866399280 , 0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(0.90617984593866399280 , -0.90617984593866399280 , 0.90617984593866399280 , 0.013299736420632648092),
            IntegrationPointType(-0.90617984593866399280 , -0.53846931010568309104 , 0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(-0.53846931010568309104 , -0.53846931010568309104 , 0.90617984593866399280 , 0.05427649123462815748),
            IntegrationPointType(0 , -0.53846931010568309104 , 0.90617984593866399280 , 0.06451200000000000000),
            IntegrationPointType(0.53846931010568309104 , -0.53846931010568309104 , 0.90617984593866399280 , 0.05427649123462815748),
            IntegrationPointType(0.90617984593866399280 , -0.53846931010568309104 , 0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(-0.90617984593866399280 , 0 , 0.90617984593866399280 , 0.031934207352848290676),
            IntegrationPointType(-0.53846931010568309104 , 0 , 0.90617984593866399280 , 0.06451200000000000000),
            IntegrationPointType(0 , 0 , 0.90617984593866399280 , 0.07667773006934522489),
            IntegrationPointType(0.53846931010568309104 , 0 , 0.90617984593866399280 , 0.06451200000000000000),
            IntegrationPointType(0.90617984593866399280 , 0 , 0.90617984593866399280 , 0.031934207352848290676),
            IntegrationPointType(-0.90617984593866399280 , 0.53846931010568309104 , 0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(-0.53846931010568309104 , 0.53846931010568309104 , 0.90617984593866399280 , 0.05427649123462815748),
            IntegrationPointType(0 , 0.53846931010568309104 , 0.90617984593866399280 , 0.06451200000000000000),
            IntegrationPointType(0.53846931010568309104 , 0.53846931010568309104 , 0.90617984593866399280 , 0.05427649123462815748),
            IntegrationPointType(0.90617984593866399280 , 0.53846931010568309104 , 0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(-0.90617984593866399280 , 0.90617984593866399280 , 0.90617984593866399280 , 0.013299736420632648092),
            IntegrationPointType(-0.53846931010568309104 , 0.90617984593866399280 , 0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(0 , 0.90617984593866399280 , 0.90617984593866399280 , 0.031934207352848290676),
            IntegrationPointType(0.53846931010568309104 , 0.90617984593866399280 , 0.90617984593866399280 , 0.026867508765371842524),
            IntegrationPointType(0.90617984593866399280 , 0.90617984593866399280 , 0.90617984593866399280 , 0.013299736420632648092)
        }};
        return s_integration_points;
    }

    static IntegrationPointsArrayType& IntegrationPoints()
    {
        return msIntegrationPoints;
    }

    std::string Info() const
    {
        std::stringstream buffer;
        buffer << "Hexadra Gauss-Legendre quadrature 5 ";
        return buffer.str();
    }
protected:

private:

   static IntegrationPointsArrayType msIntegrationPoints;

}; // Class HexahedronGaussLegendreIntegrationPoints5

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{


///@}


}  // namespace Kratos.

#endif // KRATOS_HEXAHEDRON_GAUSS_LEGENDRE_INTEGRATION_POINTS_H_INCLUDED  defined



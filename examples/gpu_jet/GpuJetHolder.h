#pragma once
#include <memory>
#include <ceres/cuda_buffer.h>
#include "ceres/jet.h"
#include <time.h>
#include <utility>
namespace ceres::examples::internal
{

struct ScalarScalarCostFunctor
{
    constexpr static const float a3 = 10;
    constexpr static const float a2 = 5;
    constexpr static const float a1 = 2;
    constexpr static const float a0 = 1;
    constexpr static const float argsDiff = 1e-13;

    template<typename T>
    JET_CUDA_DEVICE_HOST inline
    bool operator()( const T* const x , T* residual ) const
    {
        residual[ 0 ] = a3 * x[ 0 ] * x[ 0 ] * x[ 0 ] + a2 * x[ 0 ] * x[ 0 ] + a1 * x[ 0 ] + a0;
        return true;
    }

    static float analyticalDeriv(float x)
    {
        return a3 *3 *x * x  + a2 *2* x  + a1 ;
    }
};

struct VectorScalarCostFunctor
{
    constexpr static const double a3 = 10;
    constexpr static const double a2 = 5;
    constexpr static const double a1 = 2;
    constexpr static const double a0 = 1;
    constexpr static const double argsDiff = 1e-13;

    template<typename T>
    JET_CUDA_DEVICE_HOST inline
    bool operator()( const T* const x , T* residual ) const
    {
        //printf( "functor 1\n" );
        residual[ 0 ] = a3 * x[ 0 ] * x[ 0 ] * x[ 0 ]
            + a2 * x[ 0 ] * x[ 0 ]
            + a1 * x[ 0 ]
            + a0
            +( a2 - a3 ) * x[ 1 ] * x[ 1 ]
            + ( a3 - a1 ) * x[ 1 ] * x[ 1 ] * x[ 0 ]
            + ( a1 + a3 )
            * sin(
                x[ 0 ]
                + x[ 1 ]
            )
            * cos( x[ 0 ] )
            ;
//        residual[ 0 ] = a3 * x[ 0 ];
        //printf( "functor 2\n" );
        return true;
    }

};

struct VectorToVectorCostFunctor
{
    constexpr static const double r = 10;
    constexpr static const double a = 0.5;

    template<typename T>
    JET_CUDA_DEVICE_HOST inline
    bool operator()( const T* const x , T* residual ) const
    {
        //[x,y]-> ((x-cx)-r)^2 + ((y-cy)-r)^2
        /*
           |      |  
           | r[0] |     x[0]*x[0] + sin(x[1]-a*x[0])      
           | r[1] |     x[1]*x[1] + cos(x[1]-a*x[0])
           |      |  



        */

        residual[ 0 ] = x[ 0 ] * x[ 0 ] + sin( x[ 1 ] - a * x[ 0 ] );
        residual[ 1 ] = x[ 1 ] * x[ 0 ] + cos( x[ 1 ] - a * x[ 0 ] );
//        residual[ 0 ] = a3 * x[ 0 ];
        //printf( "functor 2\n" );
        return true;
    }

    static Eigen::Vector4d AnalyticDiff( const Eigen::Vector2d& p )
    {
        Eigen::Vector2d v1( 2 * p( 0 ) - a * cos( p( 1 ) - a * p( 0 ) ) ,p(1)+ a * sin( p( 1 ) - a * p( 0 ) ) );
        Eigen::Vector2d v2(  cos( p( 1 ) - a * p( 0 ) ) , p( 0 ) - sin( p( 1 ) - a * p( 0 ) ) );
        Eigen::Vector4d vret;
        vret.block<2 , 1>( 0 , 0 ) = v1;
        vret.block<2 , 1>( 2 , 0 ) = v2;
        return vret;
    }

    static Eigen::Vector4d NumericalDiff( const Eigen::Vector2d& p ,double delta,const VectorToVectorCostFunctor& cf)
    {
        Eigen::Vector2d deriv1;
        Eigen::Vector2d deriv2;
        {
            Eigen::Vector2d p1 = p - Eigen::Vector2d(1,0) * delta;
            Eigen::Vector2d p2 = p + Eigen::Vector2d(1,0) * delta;
            Eigen::Vector2d val1 , val2;
            cf( p1.data( ) , val1.data( ) );
            cf( p2.data( ) , val2.data( ) );
            deriv1 = val2 - val1;
            deriv1 *= ( 1.0 / ( 2.0 * delta ) );
        }
        {
            Eigen::Vector2d p1 = p - Eigen::Vector2d(0,1) * delta;
            Eigen::Vector2d p2 = p + Eigen::Vector2d(0,1) * delta;
            Eigen::Vector2d val1 , val2;
            cf( p1.data( ) , val1.data( ) );
            cf( p2.data( ) , val2.data( ) );
            deriv2 = val2 - val1;
            deriv2 *= ( 1.0 / ( 2.0 * delta ) );
        }
        return Eigen::Vector4d( deriv1( 0 ) , deriv1( 1 ) , deriv2( 0 ) , deriv2( 1 ) );
    }
};

class GpuJetHolder
{
public:
    GpuJetHolder( );
    GpuJetHolder( size_t pointsNum );

    GpuJetHolder( const GpuJetHolder& ) = delete;
    GpuJetHolder& operator=( const GpuJetHolder& ) = delete;

    GpuJetHolder( GpuJetHolder&& ) = default;
    GpuJetHolder& operator=( GpuJetHolder&& ) = default;
    void FillData( );
    void Run( );
    void RunAndCompare( );
    using ScalarType = double;
    using JetT = ceres::Jet<ScalarType , 2>;
    using DeriveMatrix = Eigen::Matrix<decltype( std::declval<JetT>( ).a ) , -1 , -1 , Eigen::RowMajor>;

    void RunVector2VectorCPU( );
private:
    void RunInternalGPUWithSettings( clock_t& gpuDuration ,unsigned int pperThread,unsigned int NumThreadsInBlock );
    DeriveMatrix RunInternalCPU(clock_t& cpuDuration );
    void RunInternalGPU( clock_t& gpuDuration );
    using CudaJetBuffer = ceres::internal::CudaBuffer<JetT>;
    using CudaFloatBuffer = ceres::internal::CudaBuffer< ScalarType >;
    size_t _points_num;
    std::unique_ptr< ScalarType [ ]> _points;
    std::unique_ptr< ScalarType [ ]> _derives;
    std::unique_ptr<CudaJetBuffer> _pCudaBuffer;
    std::unique_ptr<CudaFloatBuffer> _devPoints;
    std::unique_ptr<CudaFloatBuffer> _devDerives;
    std::unique_ptr<ceres::internal::CudaBuffer<VectorScalarCostFunctor> > _devFunctor;
};

//template<typename > struct SumOfSequence;
//
//template<int first>

template<int NumOfOutputs , typename NumOfInputsSeq> class IndexesHolder;

template<int NumOfOutputs , int...NumOfInputs> struct IndexesHolder<NumOfOutputs , std::integer_sequence<int,NumOfInputs...> >
{
    constexpr static const int NumOfOutJets = NumOfOutputs;
    constexpr static const int NumOfInJets = sizeof...( NumOfInputs );
    constexpr static const int Dimensions = ( NumOfInputs + ... );
    using InputsSeq = std::integer_sequence<int , NumOfInputs...>;
};

template<typename,typename > class GpuJetHolder2;



template<typename Functor, int NumOfOutputs , int...NumOfInputs> class GpuJetHolder2<Functor,std::integer_sequence<int , NumOfOutputs , NumOfInputs...> >
    : public std::enable_if_t < (sizeof...( NumOfInputs ) > 0) , IndexesHolder<NumOfOutputs , std::integer_sequence<int , NumOfInputs...> > >
{
    using IdxsHolder = IndexesHolder<NumOfOutputs , std::integer_sequence<int , NumOfInputs...> >;
public:
    GpuJetHolder2(std::vector<std::unique_ptr<Functor>>&& functors)
    {

    }
private:
    
};


}
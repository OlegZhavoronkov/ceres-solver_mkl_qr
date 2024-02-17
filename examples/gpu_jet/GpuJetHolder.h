#pragma once
#include <memory>
#include <ceres/cuda_buffer.h>
#include "ceres/jet.h"
#include <time.h>
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
    using DeriveMatrix = Eigen::Matrix<decltype(std::declval<JetT>().a) , -1 , -1 , Eigen::RowMajor>;
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

}
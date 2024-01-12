#pragma once
#include <memory>
#include <ceres/cuda_buffer.h>
#include "ceres/jet.h"
namespace ceres::examples::internal
{

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
private:
    using JetT = ceres::Jet<float , 2>;
    using CudaJetBuffer = ceres::internal::CudaBuffer<JetT>;
    size_t _points_num;
    std::unique_ptr<float [ ]> _points;
    std::unique_ptr<float [ ]> _derives;
    std::unique_ptr<CudaJetBuffer> _pCudaBuffer;
};

}
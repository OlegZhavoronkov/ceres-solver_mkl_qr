#include "GpuJetHolder.h"

namespace ceres::examples::internal
{
GpuJetHolder::GpuJetHolder( )
{
    
}

GpuJetHolder::GpuJetHolder( size_t pointsNum )
    : _points_num(pointsNum)
{
    if (_points_num > 0)
    {
        
        _points.reset( new float[ 2 * _points_num ] );
        _derives.reset( new float[ 2 * _points_num ] );
        _pCudaBuffer.reset( new CudaJetBuffer( nullptr,2 * _points_num ) );
    }
}

void GpuJetHolder::FillData( )
{
    
}

void GpuJetHolder::Run( )
{
    
}

}
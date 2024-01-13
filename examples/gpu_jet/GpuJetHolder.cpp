#include "GpuJetHolder.h"
#include <fmt/format.h>
namespace ceres::examples::internal
{
namespace
{
static std::unique_ptr<ceres::internal::ContextImpl> Context = (
                                                        [] ( )->auto
                                                        {
                                                            ceres::internal::ContextImpl* pContext = new ceres::internal::ContextImpl( );
                                                            std::string msg;
                                                            pContext->InitCuda( &msg );
                                                            return std::unique_ptr<ceres::internal::ContextImpl>( pContext );
                                                        }
                                                        )( );
}

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
        _pCudaBuffer.reset( new CudaJetBuffer( Context.get() , 2 * _points_num ) );
        _devPoints.reset( new CudaFloatBuffer( Context.get() , 2 * _points_num ) );
        _devDerives.reset( new CudaFloatBuffer( Context.get() , 2 * _points_num ) );
    }
}

void GpuJetHolder::FillData( )
{
    for (size_t i = 0; i < _points_num; i++)
    {
        float* pPoint = _points.get( ) + 2 * i;
        pPoint[ 0 ] = (i * 4 + 1)*M_1_PI;
        pPoint[ 1 ] = (i * 5 + 1)*M_1_PI;
    }
}

void GpuJetHolder::Run( )
{
    clock_t this_run;
    double whole_duration_gpu{ 0 } , whole_duration_cpu{ 0 };
    RunInternal( this_run );
    whole_duration_gpu += this_run*1000.0/CLOCKS_PER_SEC;
    RunInternalCPU( this_run );
    whole_duration_cpu += this_run*1000.0/CLOCKS_PER_SEC;
    fmt::print( "cpu {} ms gpu {} ms\n" , whole_duration_cpu , whole_duration_gpu );
}

void GpuJetHolder::RunInternalCPU( clock_t& cpuDuration )
{
    std::unique_ptr<float [ ]> cpu_deriv(new float[2*_points_num]);
    auto cpu_start = clock( );
    std::vector<JetT> jet_args( _points_num *2);
    std::vector<JetT> jet_res( _points_num );
    for (size_t i = 0; i < _points_num; i++)
    {
        jet_args[ 2 * i ] = JetT( _points[ i * 2 ] , 0 );
        jet_args[ 2 * i + 1 ] = JetT( _points[ i * 2 + 1 ] , 1 );
    }
    ScalarScalarCostFunctor cf;
    for (size_t i = 0; i < _points_num; i++)
    {
        cf( &jet_args[ 2*i ] , &jet_res[i] );
    }
    for (size_t i = 0; i < _points_num; i++)
    {
        cpu_deriv[ 2 * i ] = jet_res[ i ].v[ 0 ];
        cpu_deriv[ 2*i +1] = jet_res[ i ].v[ 1 ];
    }
    cpuDuration = clock() - cpu_start;
}

}
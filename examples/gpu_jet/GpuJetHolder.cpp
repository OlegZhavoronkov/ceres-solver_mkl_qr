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
        _devDerives.reset( new CudaFloatBuffer( Context.get( ) , 2 * _points_num ) );
        _devFunctor.reset( new ceres::internal::CudaBuffer<VectorScalarCostFunctor>( Context.get( ) , 1 ) );
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
    RunInternalGPU( this_run );
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
    VectorScalarCostFunctor cf;
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

void GpuJetHolder::RunInternalGPU( clock_t& gpuDuration )
{
    clock_t min_clock = std::numeric_limits<clock_t>::max( );
    int best_time_ppthread;
    int best_time_numThreads;
    for (int i = 1; i <= 64; i++)
    {
        for (int numThreads = 32; numThreads <= 1024; numThreads += 8)
        {
            clock_t curr_clock_summ{ 0 };
            unsigned int passes_num = 20;
            for (unsigned int j = 0; j < passes_num; j++)
            {
                clock_t curr_clock;
                RunInternalGPUWithSettings( curr_clock , i , numThreads );
                curr_clock_summ += curr_clock;
            }
            curr_clock_summ = ( clock_t ) ( curr_clock_summ / (1.0*passes_num) );
            if (min_clock > curr_clock_summ)
            {
                best_time_ppthread = i;
                best_time_numThreads = numThreads;
                min_clock = curr_clock_summ;
            }
            //fmt::print( "pperthread {} numThreads {} {} msec\n" ,i,numThreads, ( curr_clock_summ * 1000.0 ) / CLOCKS_PER_SEC );
        }
    }
    gpuDuration = min_clock;
    fmt::print( "best time {} msec pperthread {} numThreads {}\n" , ( min_clock * 1000.0 ) / CLOCKS_PER_SEC , best_time_ppthread , best_time_numThreads );
}

}
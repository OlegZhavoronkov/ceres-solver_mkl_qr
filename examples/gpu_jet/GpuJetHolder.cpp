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
        
        _points.reset( new  GpuJetHolder::ScalarType[ 2 * _points_num ] );
        _derives.reset( new GpuJetHolder::ScalarType[ 2 * _points_num ] );
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
        GpuJetHolder::ScalarType* pPoint = _points.get( ) + 2 * i;
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

void GpuJetHolder::RunAndCompare( )
{
    clock_t this_run_cpu{},this_run_gpu{};
    DeriveMatrix cpu_derives = RunInternalCPU( this_run_cpu );
    DeriveMatrix gpu_derives = RunInternalGPUWithSettings( this_run_gpu , 1 , 256 );
    fmt::print( "cpu_derives {0} {1}\n" , cpu_derives.rows( ) , cpu_derives.cols( ) );
    for (int i = 0; i < cpu_derives.rows( ) - 1; i += 1000)
    {
        fmt::print( "{0} cpu {1} {2} gpu {3} {4}\n" , i , cpu_derives( i , 0 ) , cpu_derives( i , 1 ) , gpu_derives( i , 0 ) , gpu_derives( i , 1 ) );
    }
}

GpuJetHolder::DeriveMatrix GpuJetHolder::RunInternalCPU( clock_t& cpuDuration )
{
    using scalarType = decltype( std::declval<JetT>( ).a );
    std::unique_ptr<scalarType [ ]> cpu_deriv(new scalarType[2*_points_num]);
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
    cpuDuration = clock( ) - cpu_start;
    DeriveMatrix ret = DeriveMatrix::Map( cpu_deriv.get( ) , _points_num , 2 ).eval( );
    return ret;
}

void GpuJetHolder::RunInternalGPU( clock_t& gpuDuration )
{
    clock_t min_clock = std::numeric_limits<clock_t>::max( );
    float min_clock_float = std::numeric_limits<float>::max( );
    int best_time_ppthread;
    int best_time_numThreads;
    for (unsigned int i = 1; i <= 64; i++)
    {
        for (unsigned int numThreads = 32; numThreads <= 1024; numThreads += 8)
        {
            clock_t curr_clock_summ{ 0 };
            unsigned int passes_num = 20;
            for (unsigned int j = 0; j < passes_num; j++)
            {
                clock_t curr_clock;
                RunInternalGPUWithSettings( curr_clock , i , numThreads );
                curr_clock_summ += curr_clock;
            }
            float curr_mean_clock_summ = ( clock_t ) ( curr_clock_summ / (1.0*passes_num) );
            if (min_clock_float > curr_mean_clock_summ)
            {
                best_time_ppthread = i;
                best_time_numThreads = numThreads;
                min_clock_float = curr_mean_clock_summ;
            }
            //fmt::print( "pperthread {} numThreads {} {} msec\n" ,i,numThreads, ( curr_clock_summ * 1000.0 ) / CLOCKS_PER_SEC );
        }
    }
    gpuDuration = static_cast<clock_t>(min_clock_float);
    fmt::print( "best time {} msec pperthread {} numThreads {}\n" , ( min_clock_float * 1000.0 ) / CLOCKS_PER_SEC , best_time_ppthread , best_time_numThreads );
}

}
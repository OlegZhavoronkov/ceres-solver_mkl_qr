#include <cuda.h>
#include <cuda_runtime.h>
#include "GpuJetHolder.h"
#include <fmt/format.h>



namespace ceres::examples::internal
{

__global__ void Kernel( const float* pData , float* derive , unsigned int NumPoints,GpuJetHolder::JetT* pJets,unsigned int pperthread )
{
    int currIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (currIdx*pperthread > (NumPoints+pperthread))
    {
        return;
    }
    for (unsigned int i = 0; i < pperthread; i++)
    {
        unsigned int pIdx = pperthread * currIdx + i;
        if (pIdx >= NumPoints)
        {
            return;
        }
        GpuJetHolder::JetT* jetArg = pJets + pIdx * 2;
        jetArg[ 0 ] = GpuJetHolder::JetT( pData[ pIdx * 2 ] , 0 );
        jetArg[ 1 ] = GpuJetHolder::JetT( pData[ pIdx * 2 +1] , 1 );
        GpuJetHolder::JetT res;
        ScalarScalarCostFunctor cf;
        cf( jetArg , &res );
        derive[ pIdx * 2 ] = res.v[ 0 ];
        derive[ pIdx * 2 +1] = res.v[ 1 ];
    }
    
}

void GpuJetHolder::RunInternal(clock_t& gpu_dur )
{
    unsigned int pperThread = 2;
    auto gpu_start = clock( );
    _devPoints->CopyFromCpu( _points.get( ) , 2 * _points_num );
    const int NumThreadsInBlock = 64;
    const int NumBlocks = ( _points_num / (pperThread*NumThreadsInBlock) ) + (_points_num % (pperThread*NumThreadsInBlock) == 0 ? 0 : 1);
    Kernel << <NumBlocks , NumThreadsInBlock >> > ( _devPoints->data( ) , _devDerives->data( ) , _points_num,_pCudaBuffer->data(),pperThread );
    auto res = ::cudaThreadSynchronize( );
    if (res != cudaError_t::cudaSuccess)
    {
        auto err_str = fmt::format( "error {} \"{}\"" , res , ::cudaGetErrorString( res ) );
        throw std::runtime_error( err_str );
    }
    _devDerives->CopyToCpu( _derives.get( ) , _devDerives->size( ) );
    gpu_dur = clock()-gpu_start;
    //for (size_t i = 0; i < _points_num; i++)
    //{
    //    float analytical = ScalarScalarCostFunctor::analyticalDeriv( _points[ 2 * i ] );
    //    fmt::print( " x {} deriv  {} analytical cpu {} diff {} \n" , _points[ 2 * i ] , _derives[ 2 * i ] , analytical , abs( analytical - _derives[ 2 * i ] ) );
    //}
}

}
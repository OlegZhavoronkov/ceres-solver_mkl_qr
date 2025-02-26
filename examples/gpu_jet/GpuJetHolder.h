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
    //constexpr static const double r = 10;
    const  double _a = 0.5;
    VectorToVectorCostFunctor(const double& a)
        : _a(a)
    {

    }
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

        residual[ 0 ] = x[ 0 ] * x[ 0 ] + sin( x[ 1 ] - _a * x[ 0 ] );
        residual[ 1 ] = x[ 1 ] * x[ 0 ] + cos( x[ 1 ] - _a * x[ 0 ] );
//        residual[ 0 ] = a3 * x[ 0 ];
        //printf( "functor 2\n" );
        return true;
    }

    static Eigen::Vector4d AnalyticDiff( const Eigen::Vector2d& p,double a )
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

class GpuJetHolder2Root
{
public:
    GpuJetHolder2Root( ceres::internal::ContextImpl* pImpl );
protected:
    void InitFunctorBufferOnGPU( size_t object_size , size_t objects_num );
    void InitPointsBufferOnGpu( size_t pointsNum , size_t pointSize );
    void TransferPointsToGpu( const unsigned char* ppoints , size_t  numPoints , size_t pointsize );
    template<typename Functor> void RunKernel( );
protected:
    ceres::internal::ContextImpl* _pImpl;
    using CudaBufferRaw = ceres::internal::CudaBuffer<unsigned char>;
    std::unique_ptr<CudaBufferRaw> _pCudaFunctorBufferRaw;
    std::unique_ptr<CudaBufferRaw> _pCudaPointsBufferRaw;
    std::unique_ptr<unsigned char [ ]> _pPointsBufferRaw;
};


template<typename Functor, int NumOfOutputs , int...NumOfInputs> class GpuJetHolder2<Functor,std::integer_sequence<int , NumOfOutputs , NumOfInputs...> >
    :   public std::enable_if_t < (sizeof...( NumOfInputs ) > 0) , IndexesHolder<NumOfOutputs , std::integer_sequence<int , NumOfInputs...> > >,
        public GpuJetHolder2Root
{
    using IdxsHolder = IndexesHolder<NumOfOutputs , std::integer_sequence<int , NumOfInputs...> >;
public:
    using ceresJetT = ceres::Jet < double , IdxsHolder::Dimensions>;
    using ScalarType = ceres::internal::UnderlyingScalar_t<ceresJetT>;

    
    using CudaJetBuffer = ceres::internal::CudaBuffer<ceresJetT>;
public:
    GpuJetHolder2( std::vector<std::unique_ptr<Functor>>&& functors,ceres::internal::ContextImpl* pImpl=nullptr )
        : GpuJetHolder2Root(pImpl)
        , _NumPoints(IdxsHolder::NumOfInJets*functors.size())
        , _functors( std::forward<std::vector<std::unique_ptr<Functor>>&&>(functors) )
    {
        _points.reset( new ScalarType[ _NumPoints ] );
        //_pCudaFunctorBuffer.reset( new CudaFunctorBuffer() )
    }
    void FillData( std::unique_ptr<ScalarType [ ]>&& pointsData,size_t num )
    {
        if(num!=_NumPoints)
        {
            throw std::runtime_error( "num!=_NumPoints" );
        }
        _points = std::forward<std::unique_ptr<ScalarType [ ]>&&>( pointsData );
        
    }
    void FillData( const double* p,size_t num )
    {
        if(num>_NumPoints)
        {
            throw std::runtime_error( "num>_NumPoints" );
        }
        double *pIn = const_cast<double*>( p);
        double *pSrc = _points.get();
        for (size_t n = 0; n < num; pIn++ , pSrc++)
        {
            *pSrc = *pIn;
        }
    }
    void Run()
    {
       
        if (!_pCudaFunctorBuffer)
        {
            InitFunctorBufferOnGPU( sizeof( Functor ) , _functors.size( ) );
            auto linkedBuff = CudaFunctorBuffer::Link( *_pCudaFunctorBufferRaw );
            _pCudaFunctorBuffer.reset( new CudaFunctorBuffer( std::move(linkedBuff) ) );
                
        }
        if (!_pCudaPointsBuffer)
        {
            InitPointsBufferOnGpu( _NumPoints , sizeof( ScalarType ) );
            _pCudaPointsBuffer.reset( new CudaScalarTypeBuffer( CudaScalarTypeBuffer::Link( *_pCudaPointsBufferRaw ) ) );
            TransferPointsToGpu(reinterpret_cast<unsigned char*>( _points.get()), _NumPoints , sizeof( ScalarType ));
        }
        if (_pCudaFunctorBuffer)
        {
            size_t functorsBuffSize = _functors.size( ) * sizeof( Functor );
            std::unique_ptr<unsigned char [ ]> pTempBuffer( new unsigned char[ functorsBuffSize ] );
            unsigned char* pFuncRaw = pTempBuffer.get( );
            for (int n = 0; n < _functors.size( ); n++,pFuncRaw += sizeof(Functor))
            {
                memcpy( pFuncRaw , _functors.at( n ).get( ) , sizeof( Functor ) );
            }

            _pCudaFunctorBuffer->CopyFromCpu( reinterpret_cast<Functor*>( pTempBuffer.get( )) , _functors.size( ) );
        }
        RunKernel<Functor>( );
    }

    void ExtractDerives( double* pOut , size_t num );
protected:

    const size_t _NumPoints = IdxsHolder::NumOfInJets;
    using CudaFunctorBuffer = ceres::internal::CudaBuffer<Functor>;
    using CudaScalarTypeBuffer = ceres::internal::CudaBuffer< ScalarType >;
    size_t _points_num;
    //std::unique_ptr<CudaJetBuffer> _pCudaJetBuffer;
    std::unique_ptr< ScalarType [ ]> _points;
    //std::unique_ptr< ScalarType [ ]> _derives;
    std::unique_ptr<CudaJetBuffer> _pCudaBuffer;
    std::unique_ptr<CudaFunctorBuffer> _pCudaFunctorBuffer;
    std::unique_ptr<CudaScalarTypeBuffer> _pCudaPointsBuffer;
    //std::unique_ptr<>
    //std::unique_ptr<CudaFloatBuffer> _devPoints;
    std::vector<std::unique_ptr<Functor>> _functors;
    //std::unique_ptr<CudaFloatBuffer> _devDerives;
    };


}
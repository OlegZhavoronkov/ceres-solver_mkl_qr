#include "ceres/ceres.h"
#include "glog/logging.h"
#include <fmt/format.h>
#include "ceres/jet.h"
#include <array>
#include "ceres/internal/autodiff.h"
//let's start with simple cubic function a3*x^3+a2*x^2+a1*x+a0

constexpr const double a3 = 10;
constexpr const double a2 = 5;
constexpr const double a1 = 2;
constexpr const double a0 = 1;
constexpr const double argsDiff = 1e-13;

struct ScalarCostFunctor_1arg
{
    template <typename T>
    bool operator()( const T* const x , T* residual ) const
    {
        residual[ 0 ] = a3*x[0]*x[0]*x[0]+a2*x[0]*x[0]+a1*x[0]+a0;
        return true;
    }

    static double deriv( double x )
    {
        return a3*3*x*x+a2*2*x+a1;
    }
};

struct ScalarCostFunctor_2arg
{
    template <typename T>
    bool operator()( const T* const x , T* residual ) const
    {
        residual[ 0 ] = a3 * x[ 0 ] * x[ 0 ] * x[ 0 ] + a2 * x[ 0 ] * x[ 0 ] + a1 * x[ 0 ] + a0 +
                        ( a2 - a3 ) * x[ 1 ] * x[ 1 ] + ( a3 - a1 ) * x[ 1 ] * x[ 1 ] * x[ 0 ];
        return true;
    }
    static double derive( const double* xy , double* dfdxdy )
    {
        dfdxdy[0]=a3 * 3*xy[ 0 ] * xy[ 0 ]  + a2 *2* xy[ 0 ]  + a1   
            + ( a3 - a1 ) * xy[ 1 ] * xy[ 1 ];
        dfdxdy[1] = ( a2 - a3 ) * 2*xy[ 1 ]  + ( a3 - a1 )*2 * xy[ 1 ]  * xy[ 0 ];
    }
};



void one_arg_cpu( )
{
    using jet_1 = ceres::Jet<double , 1>;
    using functor = ScalarCostFunctor_1arg;
    functor cf;
    
    std::array<double , 3> args , residual;
    args[ 1 ] = 2;
    args[ 0 ] = args[ 1 ] - argsDiff;
    args[ 2 ] = args[ 1 ] + argsDiff;
    for (size_t i = 0; i < args.size( ); i++)
    {
        cf( &args[ i ] , &residual[ i ] );
        fmt::print( "x {} residual {}\n" , args[ i ] , residual[ i ] );
    }
    double dfdx = ( residual[ 2 ] - residual[ 0 ] ) / ( args[ 2 ] - args[ 0 ] );
    double analytical = functor::deriv(args[ 1 ]);
    fmt::print( "dfdx numerical {} analytical {} diff {}\n" , dfdx , analytical , abs( analytical - dfdx ) );
    std::array<jet_1 , 3> jet_args;
    std::array<jet_1 , 3> jet_ress;
    for (size_t i = 0; i < args.size(); i++)
    {
        jet_args[ i ] = jet_1( args[ i ] , 0 );
    }
    for (size_t i = 0; i < args.size(); i++)
    {
        cf( &jet_args[ i ] , &jet_ress[ i ] );
        auto analytical = functor::deriv( args[ i ] );
        fmt::print( "dfdx jet {} analytical {} diff {}\n" , jet_ress[ i ].v[ 0 ] , analytical , abs( jet_ress[ i ].v[ 0 ] - analytical ) );
    }
}

void _2_args_cpu( )
{
    using functor = ScalarCostFunctor_2arg;
    using JetT = ceres::Jet<double , 2>;
    constexpr const int NPoints = 3;
    std::array<double , 2 * NPoints> points_data;
    functor cf;
    for (int i = 0; i < NPoints; i++)
    {
        double* pPoint = &points_data[ 2 * i ];
        pPoint[ 0 ] = i + 1;
        pPoint[ 1 ] = 2 * i + 1;
    }
    std::array<JetT , 2 * NPoints> jet_args;
    std::array<JetT , NPoints> jet_ress;
    for (int i = 0; i < NPoints; i++)
    {
        double* pPoint = &points_data[ 2 * i ];
        pPoint[ 0 ] = i + 1;
        pPoint[ 1 ] = 2 * i + 1;
        jet_args[ i * 2 ] = JetT( pPoint[ 0 ] , 0 );
        jet_args[ i * 2 + 1 ] = JetT( pPoint[ 1 ] , 1 );
    }
    for (int i = 0; i < NPoints; i++)
    {
        JetT* pJet = &jet_args[ i * 2 ];
        cf( pJet , &jet_ress[ i ] );
    }
    for (int i = 0; i < NPoints; i++)
    {
        const double* pPoint = &points_data[ 2 * i ];
        const JetT& pres = jet_ress[ i ];
        double analytical[ 2 ];
        functor::derive( pPoint , analytical );
        fmt::print( "point {} {} dfdx {} dfdy {} analytical dfdx {} dfdy {}\n" , pPoint[ 0 ] , pPoint[ 1 ] , pres.v[ 0 ] , pres.v[ 1 ],analytical[0],analytical[1] );
    }
}

int main( int argc , char** argv )
{
    GFLAGS_NAMESPACE::ParseCommandLineFlags( &argc , &argv , true );
    google::InitGoogleLogging( argv[ 0 ] );
    //one_arg_cpu( );
    _2_args_cpu( );
    return 0;
}
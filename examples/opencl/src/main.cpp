#include <absl/log/log.h>
#include <absl/log/initialize.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencl_jet/graph_node.h>

using namespace opencl_jet;

constexpr const double a3 = 10;

struct ScalarCostFunctor_quadric
{
    template <typename T>
    bool operator()( const T* const x , T* residual ) const
    {
        residual[ 0 ] = a3 * x[ 0 ] * x[ 0 ];
        return true;
    }

    static double deriv( double x )
    {
        return a3 * 2 * x;
    }
};

void quadric_pass( )
{
    int cols = 10;
    Eigen::Matrix<double , 2 , -1 , Eigen::RowMajor> data;
    data.setConstant( Eigen::NoChange , cols , 0 );
    for (int i = 0; i < 10; i++)
    {
        data( 0 , i ) = i-5;
    }
    ScalarCostFunctor_quadric func = {};
    for (int i = 0; i < 10; i++)
    {
        func( &data( 0 , i ) , &data( 1 , i ) );
    }
    OpenCLGraphNode<double> node , res;
    func( &node , &res );
}

int main( int argc , char** argv )
{
    //GFLAGS_NAMESPACE::ParseCommandLineFlags( &argc , &argv , true );
    absl::ParseCommandLine(argc, argv);
    //absl::ParseFlag( &argc , &argv , true );
    absl::InitializeLog( );
    quadric_pass( );
    return 0;
}
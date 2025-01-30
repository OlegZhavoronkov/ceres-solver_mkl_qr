#include <absl/log/log.h>
#include <absl/log/initialize.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencl_jet/graph_node.h>
#include <array>
#include "ceres/ceres.h"
#include "ceres/jet.h"
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
    OpenCLGraph graph;
    OpenCLGraphNode<double> node(&graph) , res(&graph);
    func( &node , &res );
}

template<typename T> using TracedEigenVector = Eigen::Matrix< OpenCLGraphNode< T > , 3 , 1>;
template<typename T> using TracedEigenMatrix = Eigen::Matrix< OpenCLGraphNode< T > , 3 , 3 , Eigen::RowMajor>;
void check_eigen_ctors()
{
    OpenCLGraph graph;
    OpenCLGraphNode<double> node( &graph ) , res( &graph );
    OpenCLGraphNode<double> node1( 0 );
    OpenCLGraphNode<int> node2( 0 );
}

void quadric_EigenPass( )
{
    int cols = 10;
    
    EigenScalarCtorWorkaround arg( 0 );
    
    //OpenCLGraphNode<double>
    //ScalarCostFunctor_quadric func = {};
    {
        OpenCLGraph graph;
        std::array<OpenCLGraphNode<double> , 3  > arr( { OpenCLGraphNode<double>{&graph,0},OpenCLGraphNode<double>{&graph,0},OpenCLGraphNode<double>{&graph,0} } );
        TracedEigenVector<double> vec = TracedEigenVector<double>::Map( arr.data( ) );
    }
    {
        OpenCLGraph graph;
        std::array<OpenCLGraphNode<double> , 9  > arr_intrinsics( { OpenCLGraphNode<double>{&graph,300},OpenCLGraphNode<double>{&graph,0},OpenCLGraphNode<double>{&graph,150},
                                                                        OpenCLGraphNode<double>{&graph,0},OpenCLGraphNode<double>{&graph,300},OpenCLGraphNode<double>{&graph,150},
                                                                        OpenCLGraphNode<double>{&graph,0},OpenCLGraphNode<double>{&graph,0},OpenCLGraphNode<double>{&graph,1}
         } );
        TracedEigenMatrix<double> intrinsics = TracedEigenMatrix<double>::Map( arr_intrinsics.data( ) );
        auto rev_intrinsics = intrinsics.inverse( ).eval( );
    }
    //auto reversed = vec.dot( vec );
}





void jet_tryout( )
{
    using OneDimTracedJet = TracedJet<double , 1>;
    using JetElem = TracedJetElem<double , 1>;
    OpenCLGraph graph;
    OneDimTracedJet argjet1( JetElem(&graph),0 );
    OneDimTracedJet argjet2( JetElem(&graph),0 );
    OneDimTracedJet argjet3( JetElem(&graph),0 );
    OneDimTracedJet resjet( JetElem(&graph),0 );
    ( void ) ( argjet1 + argjet2 );
    ( void ) ( argjet2 * argjet3 );
    ( void ) ( 20.0 * argjet3 );
    ( void ) ( 20.0 + argjet3 );
    ( void ) ( 20.0 - argjet3 );
    ( void ) ( argjet3 -20.0);
    ( void ) ( 20.0 / argjet3 );
    ( void ) ( argjet3 / 20.0 );
    ( void ) ( argjet3 / argjet3);
    argjet2 /= 10.0;
    //ScalarCostFunctor_quadric scf;
    //scf( &argjet , &resjet );
}

int main( int argc , char** argv )
{
    //GFLAGS_NAMESPACE::ParseCommandLineFlags( &argc , &argv , true );
    absl::ParseCommandLine(argc, argv);
    //absl::ParseFlag( &argc , &argv , true );
    absl::InitializeLog( );
//    quadric_pass( );
    //quadric_EigenPass( );
    //check_eigen_ctors( );
    jet_tryout( );
    return 0;
}
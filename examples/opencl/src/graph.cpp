#include <opencl_jet/graph.h>

namespace opencl_jet
{

OpenCLGraph::OpenCLGraph( )
    : _lastIdx(0)
{
    
}

int OpenCLGraph::NewId( )
{
    return _lastIdx++;
}

OpenCLGraph& OpenCLGraph::PushNode( int /*idx*/ , const std::string& /*operation*/ )
{
    return *this;
}

}
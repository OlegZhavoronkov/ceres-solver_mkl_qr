#pragma once
#include <atomic>
#include <string>

namespace opencl_jet
{

class OpenCLGraph
{
public:
    OpenCLGraph( );

    OpenCLGraph( const OpenCLGraph& ) = delete;
    OpenCLGraph& operator=( const OpenCLGraph& ) = delete;

    OpenCLGraph( OpenCLGraph&& ) = default;
    OpenCLGraph& operator=( OpenCLGraph&& ) = default;
    int NewId( );
    OpenCLGraph& PushNode( int idx , const std::string& operation );
private:
    std::atomic_int _lastIdx;
};

}
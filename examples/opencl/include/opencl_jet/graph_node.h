#pragma once
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <opencl_jet/graph.h>
namespace opencl_jet
{



template<typename ScalarType> class OpenCLGraphNode
{
public:
    using ThisType = OpenCLGraphNode<ScalarType>;
public:
    OpenCLGraphNode( ScalarType val = {} )
        :   _val(val),
        _pGraph( nullptr ) ,
        _graphIdx(-1)
    { }
    OpenCLGraphNode( OpenCLGraph* pGraph , ScalarType val = {} )
        :   _val( val ) ,
            _pGraph( pGraph ) ,
            _graphIdx(pGraph!=nullptr ? pGraph->NewId() : -1)
    { }

    OpenCLGraphNode( const OpenCLGraphNode& ) = delete;
    OpenCLGraphNode& operator=( const OpenCLGraphNode& ) = delete;

    OpenCLGraphNode( OpenCLGraphNode&& other )
        :   _val( std::move( other._val) ) ,
            _pGraph( other._pGraph ) ,
            _graphIdx(other._graphIdx)
    {
        other._pGraph = nullptr;
        other._graphIdx = -1;
    }

    OpenCLGraphNode& operator=( OpenCLGraphNode&& other)
    {
        std::swap( other._val , _val );
        std::swap( other._pGraph , _pGraph );
        std::swap( other._graphIdx , _graphIdx );
        return *this;
    }

public:
    //unary operators
    ThisType& operator+=( const ThisType& y )
    {
        fmt::print( "dd[{0}]+=dd[{1}]\n" , _graphIdx , y._graphIdx );
        _val += y._val;
        return *this;
    }

    ThisType& operator-=( const ThisType& y )
    {
        fmt::print( "dd[{0}]-=dd[{1}]\n" , _graphIdx , y._graphIdx );
        _val -= y._val;
        return *this;
    }

    ThisType& operator*=( const ThisType& y )
    {
        fmt::print( "dd[{0}]*=dd[{1}]\n" , _graphIdx , y._graphIdx );
        _val *= y._val;
        return *this;
    }

    ThisType& operator/=( const ThisType& y )
    {
        fmt::print( "dd[{0}]/=dd[{1}]\n" , _graphIdx , y._graphIdx );
        _val /= y._val;
        return *this;
    }

    ThisType& operator+=( const ScalarType& s )
    {
        fmt::print( "dd[{0}]+={1}\n" , _graphIdx , s );
        _val += s;
        return this;
    }

    ThisType& operator-=( const ScalarType& s )
    {
        fmt::print( "dd[{0}]-={1}\n" , _graphIdx , s );
        _val -= s;
        return *this;
    }

    ThisType& operator*=( const ScalarType& s )
    {
        fmt::print( "dd[{0}]*={1}\n" , _graphIdx , s );
        _val *= s;
        return *this;
    }

    ThisType& operator/=( const ScalarType& s )
    {
        fmt::print( "dd[{0}]/={1}\n" , _graphIdx , s );
        _val /= s;
        return *this;
    }
    //unary +
    const ThisType&  operator+( const ThisType& t )
    {
        return t;
    }
    //unary -
    ThisType  operator-( const ThisType& t )
    {
        auto ret = ThisType( t._pGraph , -t._val );
        fmt::print( "dd[{0}]= -dd[{1}]\n" , ret._graphIdx , t._graphIdx );    
        return ret;
    }
public:
    template<typename T> friend OpenCLGraphNode<T> operator+( const OpenCLGraphNode<T>& , const OpenCLGraphNode<T>& );
    template<typename T> friend OpenCLGraphNode<T> operator-( const OpenCLGraphNode<T>& , const OpenCLGraphNode<T>& );
    template<typename T> friend OpenCLGraphNode<T> operator*( const OpenCLGraphNode<T>& , const OpenCLGraphNode<T>& );
    template<typename T> friend OpenCLGraphNode<T> operator/( const OpenCLGraphNode<T>& , const OpenCLGraphNode<T>& );

    template<typename T> friend OpenCLGraphNode<T> operator+( const T& , const OpenCLGraphNode<T>& );
    template<typename T> friend OpenCLGraphNode<T> operator-( const T& , const OpenCLGraphNode<T>& );
    template<typename T> friend OpenCLGraphNode<T> operator*( const T& , const OpenCLGraphNode<T>& );
    template<typename T> friend OpenCLGraphNode<T> operator/( const T& , const OpenCLGraphNode<T>& );

    template<typename T> friend OpenCLGraphNode<T> operator+( const OpenCLGraphNode<T>& , const T&  );
    template<typename T> friend OpenCLGraphNode<T> operator-( const OpenCLGraphNode<T>& , const T&  );
    template<typename T> friend OpenCLGraphNode<T> operator*( const OpenCLGraphNode<T>& , const T&  );
    template<typename T> friend OpenCLGraphNode<T> operator/( const OpenCLGraphNode<T>& , const T&  );

private:
    ScalarType _val;
    OpenCLGraph* _pGraph;
    int _graphIdx;
};


template <typename T >
OpenCLGraphNode<T> operator+( const OpenCLGraphNode<T>& f , const OpenCLGraphNode<T>& g )
{
    auto* pGraph = f._pGraph == nullptr ? g._pGraph : f._pGraph;
    return OpenCLGraphNode<T>( pGraph , f._val + g._val );
}

// Binary + with a scalar: x + s
template <typename T >
OpenCLGraphNode<T> operator+( const OpenCLGraphNode<T>& f , T s )
{
  return OpenCLGraphNode<T>(f._pGraph ,f._val + s);
}

// Binary + with a scalar: s + x
template <typename T >
OpenCLGraphNode<T> operator+( T s , const OpenCLGraphNode<T>& f )
{
    return OpenCLGraphNode<T>(f._pGraph ,f._val + s);
}

template <typename T >
OpenCLGraphNode<T> operator-( const OpenCLGraphNode<T>& f , const OpenCLGraphNode<T>& g )
{
    auto* pGraph = f._pGraph == nullptr ? g._pGraph : f._pGraph;
    return OpenCLGraphNode<T>( pGraph , f._val - g._val );
}

// Binary + with a scalar: x + s
template <typename T >
OpenCLGraphNode<T> operator-( const OpenCLGraphNode<T>& f , T s )
{
  return OpenCLGraphNode<T>(f._pGraph ,f._val - s);
}

// Binary + with a scalar: s + x
template <typename T >
OpenCLGraphNode<T> operator-( T s , const OpenCLGraphNode<T>& f )
{
    return OpenCLGraphNode<T>(f._pGraph ,s-f._val);
}

template <typename T >
OpenCLGraphNode<T> operator*( const OpenCLGraphNode<T>& f , const OpenCLGraphNode<T>& g )
{
    auto* pGraph = f._pGraph == nullptr ? g._pGraph : f._pGraph;
    return OpenCLGraphNode<T>( pGraph , f._val * g._val );
}

// Binary + with a scalar: x + s
template <typename T >
OpenCLGraphNode<T> operator*( const OpenCLGraphNode<T>& f , T s )
{
  return OpenCLGraphNode<T>(f._pGraph ,f._val * s);
}

// Binary + with a scalar: s + x
template <typename T >
OpenCLGraphNode<T> operator*( const T& s , const OpenCLGraphNode<T>& f )
{
    return OpenCLGraphNode<T>(f._pGraph ,s*f._val);
}

template <typename T >
OpenCLGraphNode<T> operator/( const OpenCLGraphNode<T>& f , const OpenCLGraphNode<T>& g )
{
    auto* pGraph = f._pGraph == nullptr ? g._pGraph : f._pGraph;
    return OpenCLGraphNode<T>( pGraph , f._val / g._val );
}

// Binary + with a scalar: x + s
template <typename T >
OpenCLGraphNode<T> operator/( const OpenCLGraphNode<T>& f , T s )
{
  return OpenCLGraphNode<T>(f._pGraph ,f._val / s);
}

// Binary + with a scalar: s + x
template <typename T >
OpenCLGraphNode<T> operator/( T s , const OpenCLGraphNode<T>& f )
{
    return OpenCLGraphNode<T>(f._pGraph ,s/f._val);
}


}
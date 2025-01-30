#pragma once
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <opencl_jet/graph.h>
#include <type_traits>
#include <ceres/jet.h>
namespace opencl_jet
{

class EigenScalarCtorWorkaround
{
public:
    //EigenScalarCtorWorkaround( ) = delete;
    EigenScalarCtorWorkaround( int val )
        : _val( val )
    {
        //fmt::print( "{0} val {1}\n" , __PRETTY_FUNCTION__ , val );
    }
    EigenScalarCtorWorkaround( ) = delete;
    EigenScalarCtorWorkaround( const EigenScalarCtorWorkaround& ) = delete;
    EigenScalarCtorWorkaround& operator=( const EigenScalarCtorWorkaround& ) = delete;
    //EigenScalarCtorWorkaround& operator=( const int& val )
    //{
    //    fmt::print( "{0} val {1}\n" , __PRETTY_FUNCTION__ , val );
    //    _val = val;
    //    return *this;
    //}
    //explicit operator int( )
    //{
    //    return _val;
    //}
//private:
    int _val;
};



template<typename ScalarType> class OpenCLGraphNode
{
public:
    using ThisType = OpenCLGraphNode<ScalarType>;
public:
    OpenCLGraphNode( )
        : OpenCLGraphNode(ScalarType{0})
    { }
    OpenCLGraphNode(
        std::conditional_t< !std::is_same_v<ScalarType , int > , int, EigenScalarCtorWorkaround&& > from_zero_initializer
    )
        : _val{} ,
            _pGraph( nullptr ) ,
            _graphIdx(-1)
    {
        //fmt::print( "{0} {1}\n" , __PRETTY_FUNCTION__ , _val );
    }
    OpenCLGraphNode( ScalarType val  )
        :   _val(val),
        _pGraph( nullptr ) ,
        _graphIdx(-1)
    {
        //fmt::print( "{0} {1}\n" , __PRETTY_FUNCTION__ , _val );
    }
    OpenCLGraphNode( OpenCLGraph* pGraph , ScalarType val = {} )
        :   _val( val ) ,
            _pGraph( pGraph ) ,
            _graphIdx(pGraph!=nullptr ? pGraph->NewId() : -1)
    { }

    OpenCLGraphNode( const OpenCLGraphNode& src )
        :   _val( src._val ) ,
            _pGraph( src._pGraph ) ,
            _graphIdx( src._pGraph != nullptr ? src._pGraph->NewId() : -1 )
    {
        MakeUnaryExpression( src , "=" );
    }

    OpenCLGraphNode& operator=( const OpenCLGraphNode& other)
    {
        _val = other._val;
        _pGraph = _pGraph == nullptr ? other._pGraph : _pGraph;
        if (_graphIdx < 0 && _pGraph != nullptr)
        {
            _graphIdx = _pGraph->NewId( );
        }
        MakeUnaryExpression( other , "=" );
        return *this;
    }

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
        
        _val += y._val;
        MakeUnaryExpression( y , "+=" );
        return *this;
    }

    ThisType& operator-=( const ThisType& y )
    {
        _val -= y._val;
        MakeUnaryExpression( y , "-=" );
        return *this;
    }

    ThisType& operator*=( const ThisType& y )
    {
        
        _val *= y._val;
        MakeUnaryExpression( y , "*=" );
        return *this;
    }

    ThisType& operator/=( const ThisType& y )
    {
        _val /= y._val;
        MakeUnaryExpression( y , "/=" );
        return *this;
    }

    ThisType& operator+=( const ScalarType& s )
    {
        
        _val += s;
        MakeUnaryExpression( s , "+=" );
        return this;
    }

    ThisType& operator-=( const ScalarType& s )
    {
        
        _val -= s;
        MakeUnaryExpression( s , "-=" );
        return *this;
    }

    ThisType& operator*=( const ScalarType& s )
    {
        
        _val *= s;
        MakeUnaryExpression( s , "*=" );
        return *this;
    }

    ThisType& operator/=( const ScalarType& s )
    {
        
        _val /= s;
        MakeUnaryExpression( s , "/=" );
        return *this;
    }
    //unary +
    const ThisType&  operator+( const ThisType& t )const
    {
        return t;
    }
    //unary -
    ThisType  operator-( /*const ThisType& t*/ )const
    {
        auto ret = ThisType( _pGraph , -_val );
        ret.MakeUnaryPrefixExpression( *this , "-" );
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
private:
    std::string MakeUnaryExpression( const ThisType& other , const std::string& ops )
    {
        auto str = fmt::format( "dd[{0}] {1} dd[{2}]" , _graphIdx , ops , other._graphIdx );
        fmt::print( "{0}\n" , str );
        return str;
    }

    template<typename T> std::string MakeUnaryExpression( const T& operand , const std::string& ops )
    {
        auto str = fmt::format( "dd[{0}] {1} {2}" , _graphIdx , ops , operand );
        fmt::print( "{0}\n" , str );
        return str;
    }

    std::string MakeUnaryPrefixExpression( const ThisType& from , const std::string& ops = { "-" } )
    {
        auto str = fmt::format( "dd[{0}]= -dd[{1}]" , _graphIdx , from._graphIdx );
        fmt::print( "{0}\n" , str );
        return str;
    }

    std::string MakeBinaryExpression( const ThisType& op_l , const ThisType& op_r , const std::string& ops )
    {
        auto str = fmt::format( "dd[{0}]= dd[{1}] {2} dd[{3}]" , _graphIdx , op_l._graphIdx ,ops,op_r._graphIdx);
        fmt::print( "{0}\n" , str );
        return str;
    }

    template<typename T> std::string MakeBinaryExpression( const T& op_l , const ThisType& op_r , const std::string& ops )
    {
        auto str = fmt::format( "dd[{0}]= {1} {2} dd[{3}]" , _graphIdx , op_l ,ops,op_r._graphIdx);
        fmt::print( "{0}\n" , str );
        return str;
    }

    template<typename T> std::string MakeBinaryExpression(  const ThisType& op_l ,const T& op_r  ,const std::string& ops )
    {
        auto str = fmt::format( "dd[{0}]= dd[{1}] {2} {3}" , _graphIdx , op_l._graphIdx ,ops,op_r);
        fmt::print( "{0}\n" , str );
        return str;
    }
};


template <typename T >
OpenCLGraphNode<T> operator+( const OpenCLGraphNode<T>& f , const OpenCLGraphNode<T>& g )
{
    auto* pGraph = f._pGraph == nullptr ? g._pGraph : f._pGraph;
    OpenCLGraphNode<T> ret( pGraph , f._val + g._val );
    ret.MakeBinaryExpression( f , g , "+" );
    return ret;
}

// Binary + with a scalar: x + s
template <typename T >
OpenCLGraphNode<T> operator+( const OpenCLGraphNode<T>& f ,const T& s )
{
    OpenCLGraphNode<T> ret( f._pGraph , f._val + s );
    ret.MakeBinaryExpression( f , s , "+" );
    return ret;
}

// Binary + with a scalar: s + x
template <typename T >
OpenCLGraphNode<T> operator+(const T& s , const OpenCLGraphNode<T>& f )
{
    OpenCLGraphNode<T> ret( f._pGraph , f._val + s );
    ret.MakeBinaryExpression( s , f , "+" );
    return ret;
}

template <typename T >
OpenCLGraphNode<T> operator-( const OpenCLGraphNode<T>& f , const OpenCLGraphNode<T>& g )
{
    auto* pGraph = f._pGraph == nullptr ? g._pGraph : f._pGraph;
    OpenCLGraphNode<T> ret( pGraph , f._val - g._val );
    ret.MakeBinaryExpression( f , g , "-" );
    return ret;
}

// Binary + with a scalar: x + s
template <typename T >
OpenCLGraphNode<T> operator-( const OpenCLGraphNode<T>& f ,const T& s )
{
    OpenCLGraphNode<T> ret( f._pGraph , f._val - s );
    ret.MakeBinaryExpression( f , s , "-" );
    return ret;
}

// Binary + with a scalar: s + x
template <typename T >
OpenCLGraphNode<T> operator-(const T& s , const OpenCLGraphNode<T>& f )
{
    OpenCLGraphNode<T> ret( f._pGraph , s - f._val );
    ret.MakeBinaryExpression( s , f , "-" );
    return ret;
}

template <typename T >
OpenCLGraphNode<T> operator*( const OpenCLGraphNode<T>& f , const OpenCLGraphNode<T>& g )
{
    auto* pGraph = f._pGraph == nullptr ? g._pGraph : f._pGraph;
    OpenCLGraphNode<T> ret( pGraph , f._val * g._val );
    ret.MakeBinaryExpression( f , g , "*" );
    return ret;
}

// Binary + with a scalar: x + s
template <typename T >
OpenCLGraphNode<T> operator*( const OpenCLGraphNode<T>& f , const T& s )
{
    OpenCLGraphNode<T> ret( f._pGraph , f._val * s );
    ret.MakeBinaryExpression( f , s , "*" );
    return ret;
}

// Binary + with a scalar: s + x
template <typename T >
OpenCLGraphNode<T> operator*( const T& s , const OpenCLGraphNode<T>& f )
{
    OpenCLGraphNode<T> ret(f._pGraph ,s*f._val);
    ret.MakeBinaryExpression( s , f , "*" );
    return ret;
}



template <typename T >
OpenCLGraphNode<T> operator/( const OpenCLGraphNode<T>& f , const OpenCLGraphNode<T>& g )
{
    auto* pGraph = f._pGraph == nullptr ? g._pGraph : f._pGraph;
    OpenCLGraphNode<T> ret( pGraph , f._val / g._val );
    ret.MakeBinaryExpression( f , g , "/" );
    return ret;
}

// Binary + with a scalar: x + s
template <typename T >
OpenCLGraphNode<T> operator/( const OpenCLGraphNode<T>& f ,const T& s )
{
    OpenCLGraphNode<T> ret( f._pGraph , f._val / s );
    ret.MakeBinaryExpression( f , s , "/" );
    return ret;
}

// Binary + with a scalar: s + x
template <typename T >
OpenCLGraphNode<T> operator/(const T& s , const OpenCLGraphNode<T>& f )
{
    OpenCLGraphNode<T> ret( f._pGraph , s / f._val );
    ret.MakeBinaryExpression( s , f , "/" );
    return ret;
}

template<typename T , int Dimension> using TracedJet = ceres::Jet< OpenCLGraphNode<T> , Dimension >;
template<typename T , int Dimension> using TracedJetElem = typename TracedJet<T , Dimension>::Scalar;


template <typename T , int N>
inline TracedJet<T,N> operator*( const TracedJet<T,N>& f , T s )
{
    return  TracedJet<T,N>(f.a * s, f.v * s);
}


template <typename T , int N>
inline TracedJet<T,N> operator*( T s , const TracedJet<T,N>& f )
{
    return TracedJet<T,N>(f.a * s, f.v * s);
}


template <typename T , int N>
inline TracedJet<T,N> operator+( const TracedJet<T,N>& f , T s )
{
    return TracedJet<T,N>(f.a + s, f.v);
}

// Binary + with a scalar: s + x
template <typename T , int N>
inline TracedJet<T , N> operator+( T s , const TracedJet<T , N>& f )
{
    return TracedJet<T , N>( f.a + s , f.v );
}

template <typename T , int N>
inline TracedJet<T , N> operator-( const TracedJet<T , N>& f , T s )
{
    return TracedJet<T, N>(f.a - s, f.v);
}

// Binary - with a scalar: s - x
template <typename T , int N>
inline TracedJet<T , N> operator-( T s , const TracedJet<T , N>& f )
{
    return TracedJet<T, N>(s - f.a, -f.v);
}

// Binary / with a scalar: s / x
template <typename T , int N>
inline  TracedJet<T , N> operator/( T s , const TracedJet<T , N>& g )
{
    const  auto minus_s_g_a_inverse2 = -s / (g.a * g.a);
    return TracedJet<T, N>(s / g.a, g.v * minus_s_g_a_inverse2);
}

// Binary / with a scalar: x / s
template <typename T , int N>
inline TracedJet<T , N> operator/( const TracedJet<T , N>& f , T s )
{
    const T s_inverse = T(1.0) / s;
    return TracedJet<T, N>(f.a * s_inverse, f.v * s_inverse);
}

}
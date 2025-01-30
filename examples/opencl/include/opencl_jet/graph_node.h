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
    template<typename T> friend OpenCLGraphNode<T> operator/( const OpenCLGraphNode<T>& , const T& );

#ifndef DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY
#define DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(func) \
    template<typename T> friend OpenCLGraphNode<T> std::func(const OpenCLGraphNode<T>&)
#endif
#ifndef DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY
#define DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY(func) \
    template<typename T> friend OpenCLGraphNode<T> std::func(const OpenCLGraphNode<T>&,const OpenCLGraphNode<T>&)
#endif

/*unary*/
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY( abs  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY( acos ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY( asin ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY( atan ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY( cbrt ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY( ceil ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY( cos  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY( cosh ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(erf   ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(erfc  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(exp   ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(exp2  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(expm  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(floo  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(log   ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(log1  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(log1  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(log2  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(norm  ) ;
    //DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(signbit          );
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(sin   ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(sinh  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(sqrt  ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(tan   ) ;
    DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(tanh  ) ;

/*binary*/
    DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( atan2    );
    DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( pow      );
    DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( copysign );
    DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( fdim     );
    DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( fmax     );
    DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( fmin     );
    DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( hypot    );
/*ternary*/
    
    template<typename T> friend OpenCLGraphNode<T> std::fma( const OpenCLGraphNode<T>& , const OpenCLGraphNode<T>& , const OpenCLGraphNode<T>& );
/*boolean -dont know whether this will useful*/
    //using std::isinf            ;
    //using std::isnan            ;
    //using std::isnormal         ;
    //using std::isfinite         ;
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

    std::string MakeUnaryFunctionExpression( const ThisType& other , const std::string& func )
    {
        auto str = fmt::format( "dd[{0}] = {1} ( dd[{2}] )" , _graphIdx , func , other._graphIdx );
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

/*unary*/
using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::cbrt;
using std::ceil;
using std::cos;
using std::cosh;
#ifdef CERES_HAS_CPP17_BESSEL_FUNCTIONS
using std::cyl_bessel_j;
#endif  // CERES_HAS_CPP17_BESSEL_FUNCTIONS
using std::erf;
using std::erfc;
using std::exp;
using std::exp2;
using std::expm1;
using std::floor;
using std::fpclassify;
using std::isfinite;
using std::isinf;
using std::isnan;
using std::isnormal;
using std::log;
using std::log10;
using std::log1p;
using std::log2;
using std::norm;
using std::signbit;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

/*binary*/
using std::atan2;
using std::pow;
using std::copysign;
using std::fdim;
using std::fmax;
using std::fmin;
using std::hypot;
/*ternary*/
using std::fma;

//using std::abs;

//using std::acos;
//using std::asin;
//using std::atan;
//using std::atan2;
//using std::cbrt;
//using std::ceil;
//using std::copysign;
//using std::cos;
//using std::cosh;
//#ifdef CERES_HAS_CPP17_BESSEL_FUNCTIONS
//using std::cyl_bessel_j;
//#endif  // CERES_HAS_CPP17_BESSEL_FUNCTIONS
//using std::erf;
//using std::erfc;
//using std::exp;
//using std::exp2;
//using std::expm1;
//using std::fdim;
//using std::floor;
//using std::fma;
//using std::fmax;
//using std::fmin;
//using std::fpclassify;
//using std::hypot;
//using std::isfinite;
//using std::isinf;
//using std::isnan;
//using std::isnormal;
//using std::log;
//using std::log10;
//using std::log1p;
//using std::log2;
//using std::norm;
//using std::pow;
//using std::signbit;
//using std::sin;
//using std::sinh;
//using std::sqrt;
//using std::tan;
//using std::tanh;

}

namespace std
{
/*unary*/
#ifndef DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(func) \

template<typename T>\
opencl_jet::OpenCLGraphNode<T> func(const opencl_jet::OpenCLGraphNode<T>& arg)\
{ \
    opencl_jet::OpenCLGraphNode<T> ret( arg._pGraph , func( arg._val ) ); \
    ret.MakeUnaryFunctionExpression( arg , ##func ); \
    return ret; \
}

#endif
//template<typename T>
//opencl_jet::OpenCLGraphNode<T> abs(const opencl_jet::OpenCLGraphNode<T>& arg)
//{
//    opencl_jet::OpenCLGraphNode<T> ret( arg._pGraph , abs( arg._val ) );
//    ret.MakeUnaryFunctionExpression( arg , "abs" );
//    return ret;
//}
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION( abs )
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION( acos );
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION( asin ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION( atan ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION( cbrt ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION( ceil ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION( cos  ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION( cosh ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(erf   ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(erfc  ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(exp   ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(exp2  ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(expm  ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(floo  ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(log   ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(log1  ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(log1  ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(log2  ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(norm  ) ;
//DECLARE_OPENCLGRAPHNODE_FRIEND_UNARY(signbit          );
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(sin   ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(sinh  ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(sqrt  ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(tan   ) ;
DEFINE_GRAPH_NODE_UNARY_SPECIALIZATION(tanh  ) ;

/*binary*/
//DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( atan2    );
//DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( pow      );
//DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( copysign );
//DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( fdim     );
//DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( fmax     );
//DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( fmin     );
//DECLARE_OPENCLGRAPHNODE_FRIEND_BINARY( hypot    );
}
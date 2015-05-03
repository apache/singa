#ifndef MSHADOW_TENSOR_BASE_H
#define MSHADOW_TENSOR_BASE_H
/*!
 * \file tensor_base.h
 * \brief definitions of base types, macros functions
 *
 * \author Bing Xu, Tianqi Chen
 */
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <climits>
#include <algorithm>
// macro defintiions

/*!\brief if this macro is define to be 1, mshadow should compile without any of other libs */
#ifndef MSHADOW_STAND_ALONE
    #define MSHADOW_STAND_ALONE 0
#endif

/*! \brief whether do padding during allocation */
#ifndef MSHADOW_ALLOC_PAD
    #define MSHADOW_ALLOC_PAD true
#endif

/*! 
 * \brief x dimension of data must be bigger pad_size * ratio to be alloced padded memory, otherwise use tide allocation 
 *        for example, if pad_ratio=2, GPU memory alignement size is 32, then we will only allocate padded memory if x dimension > 64
 *        set it to 0 then we will always allocate padded memory
 */
#ifndef MSHADOW_MIN_PAD_RATIO
    #define MSHADOW_MIN_PAD_RATIO 2
#endif

#if MSHADOW_STAND_ALONE
   #define MSHADOW_USE_CBLAS 0
   #define MSHADOW_USE_MKL   0
   #define MSHADOW_USE_CUDA  0
#endif

/*! \brief use CBLAS for CBLAS */
#ifndef MSHADOW_USE_CBLAS
   #define MSHADOW_USE_CBLAS 0
#endif
/*! \brief use MKL for BLAS */
#ifndef MSHADOW_USE_MKL
   #define MSHADOW_USE_MKL   1
#endif
/*! \brief use CUDA support, must ensure that the cuda include path is correct, or directly compile using nvcc */
#ifndef MSHADOW_USE_CUDA
  #define MSHADOW_USE_CUDA   1
#endif
/*! \brief use single precition float */
#ifndef MSHADOW_SINGLE_PRECISION
  #define MSHADOW_SINGLE_PRECISION 1
#endif
/*! \brief whether use SSE */
#ifndef MSHADOW_USE_SSE
  #define MSHADOW_USE_SSE 1
#endif
/*! \brief whether use NVML to get dynamic info */
#ifndef MSHADOW_USE_NVML
  #define MSHADOW_USE_NVML 0
#endif
// SSE is conflict with cudacc
#ifdef __CUDACC__
  #undef MSHADOW_USE_SSE
  #define MSHADOW_USE_SSE 0
#endif

#if MSHADOW_USE_CBLAS
extern "C"{
    #include <cblas.h>
}
#elif MSHADOW_USE_MKL
  #include <mkl.h>
  #include <mkl_cblas.h>
  #include <mkl_vsl.h>
  #include <mkl_vsl_functions.h>
#endif

#if MSHADOW_USE_CUDA
  #include <cublas.h>
  #include <curand.h>
#endif

#if MSHADOW_USE_NVML
  #include <nvml.h>
#endif
// --------------------------------
// MSHADOW_XINLINE is used for inlining template code for both CUDA and CPU code.
#ifdef MSHADOW_XINLINE
  #error "MSHADOW_XINLINE must not be defined"
#endif
#ifdef __CUDACC__
  #define MSHADOW_XINLINE inline __attribute__((always_inline)) __device__ __host__
#else
  #define MSHADOW_XINLINE inline __attribute__((always_inline))
#endif
/*! \brief cpu force inline */
#define MSHADOW_CINLINE inline __attribute__((always_inline))

#if defined(__GXX_EXPERIMENTAL_CXX0X) || defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L
  #define MSHADOW_CONSTEXPR constexpr
#else
  #define MSHADOW_CONSTEXPR const
#endif

/*! \brief namespace for mshadow */
namespace mshadow {
    /*! \brief buffer size for each random number generator */
    const unsigned kRandBufferSize = 1000000;
    /*! \brief pi  */
    const float kPi = 3.1415926f;

#if MSHADOW_SINGLE_PRECISION
    /*! \brief type that will be used for content */
    typedef float real_t;
#else
    typedef double real_t;
#endif
    /*! \brief type that will be used for index */
    typedef unsigned index_t;
}; // namespace mshadow

namespace mshadow {
    /*! \brief namespace for operators */
    namespace op {
        // binary operator
        /*! \brief mul operator */
        struct mul{
            /*! \brief map a, b to result using defined operation */
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return a * b;
            }
        };
        /*! \brief plus operator */
        struct plus {
            /*! \brief map a, b to result using defined operation */
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return a + b;
            }
        };
        /*! \brief minus operator */
        struct minus {
            /*! \brief map a, b to result using defined operation */
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return a - b;
            }
        };
        /*! \brief divide operator */
        struct div {
            /*! \brief map a, b to result using defined operation */
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return a / b;
            }
        };
        /*! \brief get rhs */
        struct right {
            /*! \brief map a, b to result using defined operation */
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return b;
            }
        };
    }; // namespace op

    /*! \brief namespace for savers */
    namespace sv {
        /*! \brief save to saver: = */
        struct saveto {
            /*! \brief save b to a using save method */
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a  = b;
            }
            /*! \brief helper constant to use BLAS, alpha */
            MSHADOW_CONSTEXPR static real_t kAlphaBLAS = 1.0f;
            /*! \brief helper constant to use BLAS, beta */
            MSHADOW_CONSTEXPR static real_t kBetaBLAS  = 0.0f;
            /*! \brief corresponding binary operator type */
            typedef op::right OPType;
        };
        /*! \brief save to saver: += */
        struct plusto {
            /*! \brief save b to a using save method */
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a += b;
            }
            /*! \brief helper constant to use BLAS, alpha */
            MSHADOW_CONSTEXPR static real_t kAlphaBLAS = 1.0f;
            /*! \brief helper constant to use BLAS, beta */
            MSHADOW_CONSTEXPR static real_t kBetaBLAS  = 1.0f;
            /*! \brief corresponding binary operator type */
            typedef op::plus OPType;
        };
        /*! \brief minus to saver: -= */
        struct minusto {
            /*! \brief save b to a using save method */
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a -= b;
            }
            /*! \brief helper constant to use BLAS, alpha */
            MSHADOW_CONSTEXPR static real_t kAlphaBLAS = -1.0f;
            /*! \brief helper constant to use BLAS, beta */
            MSHADOW_CONSTEXPR static real_t kBetaBLAS  = 1.0f;
            /*! \brief corresponding binary operator type */
            typedef op::minus OPType;
        };
        /*! \brief multiply to saver: *= */
        struct multo {
            /*! \brief save b to a using save method */
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a *= b;
            }
            /*! \brief corresponding binary operator type */
            typedef op::mul OPType;
        };
        /*! \brief divide to saver: /= */
        struct divto {
            /*! \brief save b to a using save method */
            MSHADOW_XINLINE static void Save(real_t& a, real_t b) {
                a /= b;
            }
            /*! \brief corresponding binary operator type */
            typedef op::div OPType;
        };
    }; // namespace sv


    namespace op {
        // unary operator/ function: example
        // these operators can be defined by user, in the same style as binary and unary operator
        // to use, simply write F<op::identity>( src )
        /*! \brief identity function that maps a real number to it self */
        struct identity{
            /*! \brief map a to result using defined operation */
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return a;
            }
        };
    }; // namespace op

    /*! \brief namespace for potential reducer operations */
    namespace red {
        /*! \brief sum reducer */
        struct sum {
            /*! \brief do reduction into dst */
            MSHADOW_XINLINE static void Reduce( volatile real_t& dst,  volatile real_t src ) {
                dst += src;
            }
            /*! \brief calculate gradient of redres with respect to redsrc,  redres: reduced result, redsrc: one of reduction element */
            MSHADOW_XINLINE static real_t PartialGrad( real_t redres, real_t redsrc ) {
                return 1.0f;
            }
            /*! \brief an intial value of reducer */
            MSHADOW_CONSTEXPR static real_t kInitV = 0.0f;
        };
        /*! \brief maximum reducer */
        struct maximum {
            /*! \brief do reduction into dst */
            MSHADOW_XINLINE static void Reduce( volatile real_t& dst,  volatile real_t src ) {
                using namespace std;
                dst = max( dst, src );
            }
            /*! \brief calculate gradient of redres with respect to redsrc,  redres: reduced result, redsrc: one of reduction element */
            MSHADOW_XINLINE static real_t PartialGrad( real_t redres, real_t redsrc ) {
                return redres == redsrc ? 1.0f: 0.0f;
            }
            /*! \brief an intial value of reducer */
#if MSHADOW_SINGLE_PRECISION
            MSHADOW_CONSTEXPR static real_t kInitV = -FLT_MAX;
#else
            MSHADOW_CONSTEXPR static real_t kInitV = -DBL_MAX;
#endif
        };
    };

    /*! \brief namespace for helper utils of the project */
    namespace utils{
        /*! \brief send error message then exit */
        inline void Error( const char *msg ){
            fprintf( stderr, "Error:%s\n",msg );
            exit( -1 );
        }
        /*! \brief assert a expression is true */
        inline void Assert( bool exp ){
            if( !exp ) Error( "AssertError" );
        }
        /*! \brief assert a expression is true */
        inline void Assert( bool exp, const char *msg ){
            if( !exp ) Error( msg );
        }
        /*! \brief warning */
        inline void Warning( const char *msg ){
            fprintf( stderr, "warning:%s\n",msg );
        }
    }; // namespace utils
}; // namespace mshadow
#endif // TENSOR_BASE_H

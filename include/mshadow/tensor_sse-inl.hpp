#ifndef MSHADOW_TENSOR_SSE_INL_HPP
#define MSHADOW_TENSOR_SSE_INL_HPP
/*!
 * \file tensor_sse-inl.hpp
 * \brief support of sse2 optimization of some operations
 * \author Tianqi Chen
 */
#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include "tensor_expr.h"
#include "tensor.h"

namespace mshadow {
    /*! \brief namespace to support sse2 vectorization */
    namespace sse2{
        /*! 
         * \brief analog to cudaMallocPitch, allocate a aligned space with num_line * lspace cells
         * \param pitch output parameter, the actuall space allocated for each line
         * \param lspace number of cells required for each line
         * \param num_line number of lines to be allocated
         */
        inline void* AlignedMallocPitch( size_t &pitch, size_t lspace, size_t num_line ){
            pitch = ((lspace+15) >> 4) << 4;
            #ifdef _MSC_VER
            void * res = _aligned_malloc( pitch*num_line, 16 ); 
            #else
            #ifdef __APPLE__
            void *res = malloc( pitch * num_line );
            #else
            void * res = memalign( 16, pitch*num_line ); 
            #endif
            #endif
            utils::Assert( res != NULL, "AlignedMallocPitch failed" );
            return res;
        }
        /*! 
         * \brief free aligned space 
         * \param ptr pointer to space to be freed
         */
        inline void AlignedFree( void *ptr ){
            #ifdef _MSC_VER
            _aligned_free( ptr );
            #else
            free( ptr );
            #endif
        }
        /*! \brief check if a pointer is aligned */
        inline bool CheckAlign( size_t pitch ){
            return !(pitch & ((1<<4)-1));
        }
        /*! \brief check if a pointer is aligned */
        inline bool CheckAlign( void *ptr ){
            return CheckAlign( (size_t)ptr );
        }
        /*! 
         * \brief get upper bound of aligned index of size 
         * \param size size of the array
         * \param fsize size of float
         */
        inline index_t UpperAlign( index_t size, size_t fsize ){
            return (( (size*fsize+15) >> 4 ) << 4) / fsize;
        }
        /*! 
         * \brief get lower bound of aligned index of size 
         * \param size size of the array
         * \param fsize size of float
         */
        inline index_t LowerAlign( index_t size, size_t fsize ){
            return (( (size*fsize) >> 4 ) << 4) / fsize;
        }
    }; // namespace sse2
}; // namespace  mshadow

#if MSHADOW_USE_SSE
// sse types are not compatible with nvcc, only use them in cpu mode
#include <emmintrin.h>

namespace mshadow{
    namespace sse2{
        /*! 
         * \brief float vector real type, used for vectorization 
         * \tparam FloatType double or float
         */
        template<typename FloatType> struct FVec{};
        
        /*! \brief vector real type for float */
        template<> 
        struct FVec<float> {
        public:
            typedef __m128 DType;
            /*! \brief number of float in vector */
            const static index_t kSize = 4;
            /*! \brief data content */
            DType data_;
        public:
            /* constructors */
            FVec( void ){}
            FVec( DType data ):data_(data){}
            /* set the float */
            FVec( const float &s ){
                data_ = _mm_set1_ps( s );
            }
            /*!\brief load from pointer src */
            FVec( const float *src ){
                data_ = _mm_load_ps( src );                
            } 
        public:
            /*! \brief store data into dst space */
            inline void Store( float *dst ) const{
                return _mm_store_ps( dst, data_ );
            }
            /*! \brief sum of all content */
            inline float Sum( void ) const{
                DType ans  = _mm_add_ps( data_, _mm_movehl_ps( data_, data_ ) );
                DType rst  = _mm_add_ss( ans, _mm_shuffle_ps( ans, ans, 1 ) );
                #if defined(_MSC_VER) && ( _MSC_VER <= 1500 ) && defined(_WIN64)
                return rst.m128_f32[ 0 ];
                #else
                float rr = _mm_cvtss_f32( rst ) ;
                return rr;
                #endif
            }
        };

        /*! \brief vector real type for float */
        template<> 
        struct FVec<double> {
        public:
            typedef __m128d DType;
            /*! \brief number of float in vector */
            const static index_t kSize = 2;
            /*! \brief data content */
            DType data_;
        public:
            /* constructors */
            FVec( void ){}
            FVec( DType data ):data_(data){}
            /* set the float */
            FVec( const double &s ){
                data_ = _mm_set1_pd( s );
            }
            /*!\brief load from pointer src */
            FVec( const double *src ){
                data_ = _mm_load_pd( src );                
            } 
        public:
            /*! \brief store data into dst space */
            inline void Store( double *dst ) const{
                return _mm_store_pd( dst, data_ );
            }
            /*! \brief sum of all content */
            inline double Sum( void ) const{
                DType tmp =  _mm_add_sd( data_, _mm_unpackhi_pd( data_,data_ ) ) ;
                #if defined(_MSC_VER) && ( _MSC_VER <= 1500 ) && defined(_WIN64)
                return tmp.m128d_f64[0];
                #else
                double ans = _mm_cvtsd_f64( tmp );
                return ans;
                #endif
            }
        };
    };

    namespace sse2{
        /*! \brief sse2 operator type of certain operator */
        template<typename OP>
        struct SSEOp{
            const static bool kEnabled = false;
        };        
        template<>
        struct SSEOp<op::plus>{
            const static bool kEnabled = true;
            MSHADOW_CINLINE static FVec<float> Map( const FVec<float> &lhs, const FVec<float> &rhs ){
                return FVec<float>( _mm_add_ps( lhs.data_, rhs.data_ ) );
            }
            MSHADOW_CINLINE static FVec<double> Map( const FVec<double> &lhs, const FVec<double> &rhs ){
                return FVec<double>( _mm_add_pd( lhs.data_, rhs.data_ ) );
            }
        };
        template<>
        struct SSEOp<op::minus>{
            const static bool kEnabled = true;
            MSHADOW_CINLINE static FVec<float> Map( const FVec<float> &lhs, const FVec<float> &rhs ){
                return FVec<float>( _mm_sub_ps( lhs.data_, rhs.data_ ) );
            }
            MSHADOW_CINLINE static FVec<double> Map( const FVec<double> &lhs, const FVec<double> &rhs ){
                return FVec<double>( _mm_sub_pd( lhs.data_, rhs.data_ ) );
            }
        };
        template<>
        struct SSEOp<op::mul>{
            const static bool kEnabled = true;
            MSHADOW_CINLINE static FVec<float> Map( const FVec<float> &lhs, const FVec<float> &rhs ){
                return FVec<float>( _mm_mul_ps( lhs.data_, rhs.data_ ) );
            }
            MSHADOW_CINLINE static FVec<double> Map( const FVec<double> &lhs, const FVec<double> &rhs ){
                return FVec<double>( _mm_mul_pd( lhs.data_, rhs.data_ ) );
            }
        };
        template<>
        struct SSEOp<op::div>{
            const static bool kEnabled = true;
            MSHADOW_CINLINE static FVec<float> Map( const FVec<float> &lhs, const FVec<float> &rhs ){
                return FVec<float>( _mm_div_ps( lhs.data_, rhs.data_ ) );
            }
            MSHADOW_CINLINE static FVec<double> Map( const FVec<double> &lhs, const FVec<double> &rhs ){
                return FVec<double>( _mm_div_pd( lhs.data_, rhs.data_ ) );
            }
        };

        template<>
        struct SSEOp<op::identity>{
            const static bool kEnabled = true;
            MSHADOW_CINLINE static FVec<float> Map( const FVec<float> &src ){
                return src;
            }
            MSHADOW_CINLINE static FVec<double> Map( const FVec<double> &src ){
                return src;
            }
        };
    }; // namespace sse2
    
    namespace sse2{
        // savers to do storage
        template<typename SV, typename TFloat>
        struct Saver{
            MSHADOW_CINLINE static void Save( TFloat *dst, const FVec<TFloat> &src ){
                FVec<TFloat> lhs( dst );
                FVec<TFloat> ans = SSEOp<typename SV::OPType>::Map( lhs, src );
                ans.Store( dst );
            }
        };
        template<typename TFloat>
        struct Saver<sv::saveto,TFloat>{
            MSHADOW_CINLINE static void Save( TFloat *dst, const FVec<TFloat> &src ){
                src.Store( dst );
            }
        };        
    }; // namespace sse2
}; // namespace mshadow

namespace mshadow{
    namespace expr{
        // same as plan, but use sse2
        template<typename ExpType>
        class SSEPlan {
        public:
            /*!
             * \brief evaluate the expression at index [y][x], x will be aligned to 4
             *        to be implemented by SubType
             */
            MSHADOW_CINLINE sse2::FVec<real_t> EvalSSE( index_t y, index_t x ) const;
            MSHADOW_CINLINE real_t Eval( index_t y, index_t x ) const;
        };

        template <typename Device, int dim>
        class SSEPlan< Tensor<Device,dim> >{
        public:
            SSEPlan( const Tensor<Device,dim> &t )
                :dptr_(t.dptr),stride_(t.shape.stride_){}
            MSHADOW_CINLINE sse2::FVec<real_t> EvalSSE( index_t y, index_t x ) const{
                return sse2::FVec<real_t>( &dptr_[ y*stride_+x ] );
            }
            MSHADOW_CINLINE real_t Eval( index_t y, index_t x ) const{
                return dptr_[ y * stride_ + x ];
            }
        private:
            const real_t  *dptr_;
            index_t stride_;
        };

        template<>
        class SSEPlan<ScalarExp>{
        public:
            SSEPlan( real_t scalar ):scalar_(scalar){}
            MSHADOW_CINLINE sse2::FVec<real_t> EvalSSE( index_t y, index_t x ) const{
                return sse2::FVec<real_t>( scalar_ );
            }
            MSHADOW_CINLINE real_t Eval( index_t y, index_t x ) const{
                return scalar_;
            }
        private:
            real_t scalar_;
        };

        template<typename OP, typename TA, typename TB,int etype>
        class SSEPlan< BinaryMapExp<OP,TA,TB,etype> >{
        public:
            SSEPlan( const SSEPlan<TA> &lhs, const SSEPlan<TB> &rhs )
                :lhs_(lhs), rhs_(rhs){}
            MSHADOW_CINLINE sse2::FVec<real_t> EvalSSE( index_t y, index_t x ) const{
                return sse2::SSEOp<OP>::Map( lhs_.EvalSSE( y, x ), rhs_.EvalSSE( y, x ) );
            }
            MSHADOW_CINLINE real_t Eval( index_t y, index_t x ) const{
                return OP::Map( lhs_.Eval( y, x ), rhs_.Eval( y, x ) );
            }
        private:
            SSEPlan<TA> lhs_;
            SSEPlan<TB> rhs_;
        };

        template<typename OP, typename TA, int etype>
        class SSEPlan< UnaryMapExp<OP,TA,etype> >{
        public:
            SSEPlan( const SSEPlan<TA> &src ):src_(src){}
            MSHADOW_CINLINE sse2::FVec<real_t> EvalSSE( index_t y, index_t x ) const{
                return sse2::SSEOp<OP>::Map( src_.EvalSSE( y, x ) );
            }
            MSHADOW_CINLINE real_t Eval( index_t y, index_t x ) const{
                return OP::Map( src_.Eval( y, x ) );
            }
        private:
            SSEPlan<TA> src_;
        };

        template<typename OP, typename TA, typename TB, int etype>
        inline SSEPlan< BinaryMapExp<OP,TA,TB,etype> > MakeSSEPlan( const BinaryMapExp<OP,TA,TB,etype> &e );

        inline SSEPlan<ScalarExp> MakeSSEPlan( const ScalarExp &e ){
            return SSEPlan<ScalarExp>( e.scalar_ );
        }

        template<typename T>
        inline SSEPlan<T> MakeSSEPlan( const ContainerExp<T> &e ){
            return SSEPlan<T>( e.self() );
        }

        template<typename T,int dim>
        inline SSEPlan<T> MakeSSEPlan( const MakeTensorExp<T,cpu,dim> &e ){
            return SSEPlan<T>( e.real_self() );
        }

        template<typename OP, typename TA, int etype>
        inline SSEPlan< UnaryMapExp<OP,TA,etype> > MakeSSEPlan( const UnaryMapExp<OP,TA,etype> &e ){
            return SSEPlan< UnaryMapExp<OP,TA,etype> >( MakeSSEPlan(e.src_) );
        }

        template<typename OP, typename TA, typename TB, int etype>
        inline SSEPlan< BinaryMapExp<OP,TA,TB,etype> > MakeSSEPlan( const BinaryMapExp<OP,TA,TB,etype> &e ){
                return SSEPlan< BinaryMapExp<OP,TA,TB,etype> >( MakeSSEPlan(e.lhs_), MakeSSEPlan(e.rhs_) );
        }
    };

    namespace expr{
        /*!
         * \brief static check sse enable
         *        if a expression E can not be evaluated using sse, then kPass = false
         * \tparam Device the type of Device
         * \tparam dim dimension of the tensor
         * \tparam E expression
         */
        template<typename E>
        struct SSECheck{
            const static bool kPass = false;
        };
        template<>
        struct SSECheck<ScalarExp>{
            const static bool kPass = true;
        };
        template<int dim>
        struct SSECheck<Tensor<cpu,dim> >{
            const static bool kPass = true;
        };
        
        template<typename OP, typename TA, int etype>
        struct SSECheck<UnaryMapExp<OP,TA,etype> >{
            const static bool kPass = SSECheck<TA>::kPass && sse2::SSEOp<OP>::kEnabled;
        };
        template<typename OP, typename TA, typename TB, int etype>
        struct SSECheck< BinaryMapExp<OP,TA,TB,etype> >{
            const static bool kPass = SSECheck<TA>::kPass && SSECheck<TB>::kPass && sse2::SSEOp<OP>::kEnabled;
        }; 
    }; // namespace expr
    namespace expr{
        // check if data is aligned and allow sse operation
        template<int dim,typename E>
        struct SSEAlignCheck{
            inline static bool Check( const E &exp ){
                return false;
            }
        };
        template<int dim>
        struct SSEAlignCheck< dim, ScalarExp >{
            inline static bool Check( const ScalarExp &exp ){
                return true;
            }
        };
        template<int dim>
        struct SSEAlignCheck< dim,Tensor<cpu,dim> >{
            inline static bool Check( const Tensor<cpu,dim> &t ){
                return sse2::CheckAlign( t.dptr ) && sse2::CheckAlign( t.shape.stride_ * sizeof( real_t ) );
            }
        };
        template<int dim, typename OP, typename TA, int etype>
        struct SSEAlignCheck< dim, UnaryMapExp<OP,TA,etype> >{
            inline static bool Check( const UnaryMapExp<OP,TA,etype> &t ){
                return SSEAlignCheck<dim,TA>::Check( t.src_);
            }
        };
        template<int dim, typename OP, typename TA, typename TB, int etype>
        struct SSEAlignCheck< dim, BinaryMapExp<OP,TA,TB,etype> >{ 
            inline static bool Check( const BinaryMapExp<OP,TA,TB,etype> &t ){
                return SSEAlignCheck<dim,TA>::Check( t.lhs_ ) && 
                    SSEAlignCheck<dim,TB>::Check( t.rhs_ );
            }
        };
    }; // namespace expr

    /*! 
     * \brief use SSEPlan to compute result
     */
    template<typename SV, typename E, int dim>
    inline void MapSSEPlan(Tensor<cpu,dim> _dst, const expr::SSEPlan<E> &plan){        
        Tensor<cpu,2> dst = _dst.FlatTo2D();
        const index_t xlen = sse2::LowerAlign( dst.shape[0], sizeof(real_t) );
        for ( index_t y = 0; y < dst.shape[1]; y ++ ) {
            for( index_t x = 0; x < xlen; x += sse2::FVec<real_t>::kSize ){
                sse2::Saver<SV,real_t>::Save( &dst[y][x], plan.EvalSSE( y,x ) );
            }
            for( index_t x = xlen; x < dst.shape[0]; x ++ ){
                SV::Save( dst[y][x], plan.Eval(y,x) );
            }
        }
    }
}; // namespace mshadow
#endif // MSHADOW_USE_SSE
#endif // MSHADOW_TENSOR_SSE_INL_HPP

#ifndef MSHADOW_TENSOR_CPU_INL_HPP
#define MSHADOW_TENSOR_CPU_INL_HPP
/*!
 * \file tensor_cpu-inl.hpp
 * \brief implementation of CPU host code
 * \author Bing Xu, Tianqi Chen
 */
#include <cstring>
#include "tensor_base.h"
#include "tensor_sse-inl.hpp"

namespace mshadow {
    template<int dim>
    inline void AllocSpace(Tensor<cpu,dim> &obj, bool pad ){
        size_t pitch;
        if( pad ){
            obj.dptr = (real_t*)sse2::AlignedMallocPitch
                ( pitch, obj.shape[0] * sizeof(real_t), obj.FlatTo2D().shape[1] );
            obj.shape.stride_ = static_cast<index_t>( pitch / sizeof(real_t) );
        }else{
            obj.shape.stride_ = obj.shape[0];
            obj.dptr = (real_t*)sse2::AlignedMallocPitch
                ( pitch, obj.shape.Size() * sizeof(real_t), 1 );
        }
    }

    template<typename Device, int dim>
    inline Tensor<Device,dim> NewTensor(const Shape<dim> &shape, real_t initv, bool pad ){
        Tensor<Device, dim> obj( shape );
        AllocSpace( obj, pad );
        MapExp<sv::saveto>( obj, expr::ScalarExp( initv ) );
        return obj;
    }

    template<int dim>
    inline void FreeSpace(Tensor<cpu,dim> &obj){
        sse2::AlignedFree( obj.dptr );
        obj.dptr = NULL;
    }

    template<int dim>
    inline void Copy(Tensor<cpu,dim> _dst, const Tensor<cpu,dim> &_src ){
        utils::Assert( _dst.shape == _src.shape, "Copy:shape mismatch" );
        Tensor<cpu,2> dst = _dst.FlatTo2D();
        Tensor<cpu,2> src = _src.FlatTo2D();
        for (index_t y = 0; y < dst.shape[1]; ++y ) {
            memcpy( dst[y].dptr, src[y].dptr, sizeof(real_t) * dst.shape[0] );
        }
    }

    template<typename Saver, typename E, int dim>
    inline void MapPlan(Tensor<cpu,dim> _dst, const expr::Plan<E> &plan){
        Tensor<cpu,2> dst = _dst.FlatTo2D();
        for (index_t y = 0; y < dst.shape[1]; ++y ) {
            for (index_t x = 0; x < dst.shape[0]; ++x ) {
                // trust your compiler! -_- they will optimize it
                Saver::Save(dst[y][x], plan.Eval( y, x ) );
            }
        }
    }

    // code to handle SSE optimization
    template<bool pass_check,typename Saver, int dim, typename E, int etype>
    struct MapExpCPUEngine;
    template<typename SV, int dim, typename E, int etype>
    struct MapExpCPUEngine<false,SV,dim,E,etype>{
        inline static void Map(Tensor<cpu,dim> dst, const expr::Exp<E,etype> &exp ){
            MapPlan<SV>( dst, MakePlan( exp.self() ) );
        }
    };

    #if MSHADOW_USE_SSE
    template<typename SV, int dim, typename E, int etype>
    struct MapExpCPUEngine<true,SV,dim,E,etype>{
        inline static void Map(Tensor<cpu,dim> dst, const expr::Exp<E,etype> &exp ){
            using namespace expr;
            if( SSEAlignCheck<dim,E>::Check( exp.self() ) && SSEAlignCheck< dim,Tensor<cpu,dim> >::Check(dst) ){
                MapSSEPlan<SV>( dst, MakeSSEPlan( exp.self() ) );
            }else{
                MapPlan<SV>( dst, MakePlan( exp.self() ) );
            }
        }
    };
    #endif

    template<typename Saver, int dim, typename E, int etype>
    inline void MapExp(Tensor<cpu,dim> dst, const expr::Exp<E,etype> &exp ){
        using namespace expr;
        TypeCheckPass< TypeCheck<cpu,dim,E>::kMapPass >::Error_All_Tensor_in_Exp_Must_Have_Same_Type();
        Shape<dim> eshape = ShapeCheck<dim,E>::Check( exp.self() );
        utils::Assert( eshape[0] == 0 || eshape == dst.shape, "Assignment: Shape of Tensors in expression is not consistent with target" );
        #if MSHADOW_USE_SSE
        MapExpCPUEngine< SSECheck<E>::kPass,Saver,dim,E,etype >::Map( dst, exp );
        #else
        MapExpCPUEngine< false,Saver,dim,E,etype >::Map( dst, exp );
        #endif
    }

    template<typename Saver, typename Reducer, typename E, int etype>
    inline void MapReduceKeepLowest( Tensor<cpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale ){
        using namespace expr;
        TypeCheckPass< TypeCheck<cpu,1,E>::kRedPass >::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
        Shape<2> eshape = ShapeCheck< ExpInfo<E>::kDim, E >::Check( exp.self() ).FlatTo2D();

        utils::Assert( eshape[0] == dst.shape[0], "reduction dimension do not match" );
        utils::Assert( eshape[1] != 0, "can not reduce over empty tensor" );
        // execution
        expr::Plan<E> plan = MakePlan( exp.self() );
        for( index_t x = 0; x < eshape[0]; ++x ){
            real_t res = plan.Eval( 0, x );
            for( index_t y = 1; y < eshape[1]; ++y ){
                Reducer::Reduce( res, plan.Eval( y, x ) );
            }
            Saver::Save( dst[x], res*scale );
        }
    }

    template<typename Saver, typename Reducer, int dimkeep, typename E, int etype>
    inline void MapReduceKeepHighDim( Tensor<cpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale ){
        using namespace expr;
        TypeCheckPass< TypeCheck<cpu,dimkeep,E>::kRedPass >::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
        typedef Shape< ExpInfo<E>::kDim > EShape;
        EShape eshape = ShapeCheck< ExpInfo<E>::kDim, E >::Check( exp.self() );
        utils::Assert( eshape[dimkeep] == dst.shape[0], "reduction dimension do not match" );
        // use equvalent form
        Shape<4> pshape = Shape4( eshape.ProdShape(dimkeep+1,EShape::kMaxShape), eshape[dimkeep], 
                                  eshape.ProdShape(1,dimkeep), eshape[0] );

        // execution
        expr::Plan<E> plan = MakePlan( exp.self() );

        for( index_t c = 0; c < pshape[2]; ++c ){
            real_t res = Reducer::kInitV;
            for( index_t n = 0; n < pshape[3]; ++n ){
                real_t tres = Reducer::kInitV;
                for( index_t y = 0; y < pshape[1]; ++y ){
                    for( index_t x = 0; x < pshape[0]; ++x ){
                        Reducer::Reduce( tres, plan.Eval( (n*pshape[2] + c) * pshape[1] + y, x ) );
                    }
                }
                Reducer::Reduce( res, tres );
            }
            Saver::Save( dst[c], res*scale );
        }
    }

    inline void Softmax( Tensor<cpu,1> dst, const Tensor<cpu,1>& energy ){
        real_t mmax = energy[0];
        for( real_t x = 1; x < dst.shape[0]; ++x )
            if( mmax < energy[x] ) mmax = energy[x];
        real_t sum = 0.0f;
        for( index_t x = 0; x < dst.shape[0]; ++x ){
            dst[x] = std::exp( energy[x] - mmax );
            sum += dst[x];
        }
        for( index_t x = 0; x < dst.shape[0]; ++x ){
            dst[x] /= sum;
        }
    }
    inline void Softmax( Tensor<cpu,2> dst, const Tensor<cpu,2>& energy ){
        utils::Assert( dst.shape == energy.shape, "Softmax: shape mismatch" );
        for( index_t y = 0; y < dst.shape[1]; ++y ){
            Softmax( dst[y], energy[y] );
        }
    }
}; // namespace mshadow

#endif // TENSOR_CPU_INL_HPP

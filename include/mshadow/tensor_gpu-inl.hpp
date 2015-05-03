#ifndef MSHADOW_TENSOR_GPU_INL_HPP
#define MSHADOW_TENSOR_GPU_INL_HPP
/*!
 * \file tensor_gpu-inl.hpp
 * \brief implementation of GPU host code
 * \author Bing Xu, Tianqi Chen
 */
#include "tensor.h"

#if !(MSHADOW_USE_CUDA)
namespace mshadow {
    // do nothing if no GPU operation is involved
    inline void InitTensorEngine( int dev_id ){
    }
    inline void ShutdownTensorEngine( void ){
    }
};
#else
namespace mshadow {
    #if (MSHADOW_USE_NVML)
    inline int AutoSelectDevice(int device_count) {
        // TODO nvml device id and cuda device id are not consistent
        return 0;
    }
    #endif
    inline void InitTensorEngine(int dev_id){
        cudaDeviceProp prop;
        int device_id = 0;
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        utils::Assert(device_count > 0, "Cannot find CUDA device. Please check CUDA-Configuration");
        if (dev_id < 0) {
            #if (MSHADOW_USE_NVML)
            device_id = AutoSelectDevice(device_count);
            #endif
        } else {
            device_id = dev_id;
        }
        utils::Assert( device_id < device_count, "Incorrect Device ID" );
        utils::Assert( cudaSetDevice(device_id) == cudaSuccess, "cannot set device" );
        cudaGetDeviceProperties(&prop, device_id);
        printf("Use CUDA Device %d: %s\n", device_id, prop.name);
        cublasInit();
    }
    inline void ShutdownTensorEngine( void ){
        cublasShutdown();
    }

    template<int dim>
    inline void AllocSpace(Tensor<gpu,dim> &obj, bool pad){
        size_t pitch;
        // common choice for cuda mem align unit is 32
        if( pad && obj.shape[0] >= MSHADOW_MIN_PAD_RATIO * 32 ){
            cudaError_t err = cudaMallocPitch( (void**)&obj.dptr, &pitch, \
                                               obj.shape[0] * sizeof(real_t), obj.FlatTo2D().shape[1] );
            utils::Assert( err == cudaSuccess, cudaGetErrorString(err) );
            obj.shape.stride_ = static_cast<index_t>( pitch / sizeof(real_t) );
        }else{
            obj.shape.stride_ = obj.shape[0];
            cudaError_t err = cudaMallocPitch( (void**)&obj.dptr, &pitch, \
                                               obj.shape.Size() * sizeof(real_t), 1 );
            utils::Assert( err == cudaSuccess, cudaGetErrorString(err) );
        }
    }

    template<int dim>
    inline void FreeSpace(Tensor<gpu,dim> &obj){
        cudaFree( obj.dptr ); obj.dptr = NULL;
    }

    template<typename A,typename B, int dim>
    inline void Copy(Tensor<A,dim> _dst, Tensor<B,dim> _src, cudaMemcpyKind kind){
        utils::Assert( _dst.shape == _src.shape, "Copy:shape mismatch" );
        Tensor<A,2> dst = _dst.FlatTo2D();
        Tensor<B,2> src = _src.FlatTo2D();
        cudaError_t err = cudaMemcpy2D( dst.dptr, dst.shape.stride_ * sizeof(real_t),
                                        src.dptr, src.shape.stride_ * sizeof(real_t),
                                        dst.shape[0] * sizeof(real_t),
                                        dst.shape[1], kind );
        utils::Assert( err == cudaSuccess, cudaGetErrorString(err) );
    }
    template<int dim>
    inline void Copy(Tensor<cpu,dim> dst, const Tensor<gpu,dim> &src){
        Copy( dst, src, cudaMemcpyDeviceToHost );
    }
    template<int dim>
    inline void Copy(Tensor<gpu,dim> dst, const Tensor<gpu,dim> &src){
        Copy( dst, src, cudaMemcpyDeviceToDevice );
    }
    template<int dim>
    inline void Copy(Tensor<gpu,dim> dst, const Tensor<cpu,dim> &src){
        Copy( dst, src, cudaMemcpyHostToDevice );
    }
};

#ifdef __CUDACC__
// the following part is included only if compiler is nvcc
#include "cuda/tensor_gpu-inl.cuh"

namespace mshadow{
    template<typename Saver, typename E, int dim>
    inline void MapPlan(Tensor<gpu,dim> _dst, const expr::Plan<E> &plan){
        cuda::MapPlan<Saver>( _dst.FlatTo2D(), plan );
    }

    template<typename Saver, int dim, typename E, int etype>
    inline void MapExp(Tensor<gpu,dim> dst, const expr::Exp<E,etype> &exp ){
        using namespace expr;
        TypeCheckPass< TypeCheck<gpu,dim,E>::kMapPass >::Error_All_Tensor_in_Exp_Must_Have_Same_Type();
        Shape<dim> eshape = ShapeCheck<dim,E>::Check( exp.self() );
        utils::Assert( eshape[0] == 0 || eshape == dst.shape, "Assignment: Shape of Tensors in expression is not consistent with target" );
        MapPlan<Saver>( dst, MakePlan( exp.self() ) );
    }

    template<typename Saver, typename Reducer, typename E, int etype>
    inline void MapReduceKeepLowest( Tensor<gpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale ){
        using namespace expr;
        TypeCheckPass< TypeCheck<gpu,1,E>::kRedPass >::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
        Shape<2> eshape = ShapeCheck< ExpInfo<E>::kDim, E >::Check( exp.self() ).FlatTo2D();

        utils::Assert( eshape[0] == dst.shape[0], "reduction dimension do not match" );
        utils::Assert( eshape[1] != 0, "can not reduce over empty tensor" );
        cuda::MapReduceKeepLowest<Saver,Reducer>( dst, MakePlan( exp.self() ), scale, eshape );
    }

    template<typename Saver, typename Reducer, int dimkeep, typename E, int etype>
    inline void MapReduceKeepHighDim( Tensor<gpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale ){
        using namespace expr;
        TypeCheckPass< TypeCheck<gpu,dimkeep,E>::kRedPass >::Error_TypeCheck_Not_Pass_For_Reduce_Exp();
        typedef Shape< ExpInfo<E>::kDim > EShape;
        EShape eshape = ShapeCheck< ExpInfo<E>::kDim, E >::Check( exp.self() );
        utils::Assert( eshape[dimkeep] == dst.shape[0], "reduction dimension do not match" );
        // use equvalent form
        Shape<4> pshape = Shape4( eshape.ProdShape(dimkeep+1,EShape::kMaxShape), eshape[dimkeep],
                                  eshape.ProdShape(1,dimkeep), eshape[0] );
        // call equavalent map red dim 2
        cuda::MapReduceKeepDim2<Saver,Reducer>( dst, MakePlan( exp.self() ), scale, pshape );
    }

    inline void Softmax( Tensor<gpu,2> dst, const Tensor<gpu,2>& src ){
        cuda::Softmax( dst, src );
    }
}; // namespace mshadow

#endif // __CUDACC__

#endif // MSHADOW_USE_CUDA
#endif // TENSOR_GPU_INL_HPP

#ifndef MSHADOW_TENSOR_GPU_INL_CUH
#define MSHADOW_TENSOR_GPU_INL_CUH
/*!
 * \file tensor_gpu-inl.cuh
 * \brief implementation of GPU code using CUDA
 * \author Bing Xu, Tianqi Chen
 */
#include "../tensor.h"
#include "cuda_reduce.cuh"

namespace mshadow{
    namespace cuda{
        #ifndef __CUDA_ARCH__
        #warning "__CUDA_ARCH__ is not defined, I will assume compiling with CUDA verion greater than 2.0"
        #endif
        /* load unit for memory access */
        #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200
        const int kMemUnitBits = 5;
        const int kMaxThreadsPerBlock = 1024;
        #else
        const int kMemUnitBits = 4;
        const int kMaxThreadsPerBlock = 512;
        #endif
        /*! \brief number of units that can do synchronized update, half warp size */
        const int kMemUnit     = 1 << kMemUnitBits;
        /*! \brief mask that could be helpful sometime */
        const int kMemUnitMask = kMemUnit - 1;
        /*! \brief suggested thread number(logscale) for mapping kernel */
        const int kBaseThreadBits = 8;
        /*! \brief suggested thread number for mapping kernel */
        const int kBaseThreadNum  = 1 << kBaseThreadBits;
        /*! \brief maximum value of grid */
        const int kMaxGridNum     = 65535;
        /*! \brief suggested grid number for mapping kernel */
        const int kBaseGridNum    = 1024;
        
        /*! \brief get align stride for given size in x dimension */
        inline index_t GetAlignStride( index_t xsize, index_t xstride ){ 
            if( (xstride & (kMemUnit-1)) == 0 ){
                return ( (xsize  + kMemUnit - 1) >> kMemUnitBits) << kMemUnitBits;
            }else{
                // if originally space is not aligned, no necessary to to alligned thread allocation
                return xsize;
            }
        }
        inline void CheckLaunchParam( dim3 dimGrid, dim3 dimBlock, const char *estr = "" ){
            if( dimBlock.x*dimBlock.y*dimBlock.z > (unsigned)kMaxThreadsPerBlock ||
                dimGrid.x > 65535 || dimGrid.y > 65535 ){
                fprintf( stderr, "%s[%u,%u,%u]:", estr, dimBlock.x, dimBlock.y, dimBlock.z );
                utils::Error( "too large launch parameter\n");
            } 
        }        
    };

    namespace cuda {
        template<typename Saver, typename Plan, int block_dim_bits>
        __device__ void MapPlanProc( Tensor<gpu,2> dst, const index_t xstride, const Plan exp, int block_idx ){
            const index_t tid = (block_idx << block_dim_bits) + threadIdx.x;
            const int y   = tid / xstride;
            const int x   = tid % xstride;
            if (y < dst.shape[1] && x < dst.shape[0]) {
                Saver::Save(dst[y][x], exp.Eval(y,x));
            }
        }
        template<typename Saver, typename Plan, int block_dim_bits>
        __global__ void MapPlanKernel( Tensor<gpu,2> dst, const index_t xstride, const Plan exp ){
            MapPlanProc<Saver, Plan,block_dim_bits>( dst, xstride, exp, blockIdx.x );
        }
        template<typename Saver, typename Plan, int block_dim_bits, int grid_size>
        __global__ void MapPlanLargeKernel( Tensor<gpu,2> dst, const index_t xstride, const Plan exp, int repeat ){
            for( int i = 0; i < repeat; ++i ){
                MapPlanProc<Saver, Plan,block_dim_bits>( dst, xstride, exp, blockIdx.x + i*grid_size );
            }
        }        
        
        template<typename Saver, typename E>
        inline void MapPlan( Tensor<gpu,2> dst, const expr::Plan<E> &plan ){
            const index_t xstride = GetAlignStride( dst.shape[0], dst.shape.stride_ );
            const int num_block = ( dst.shape[1]*xstride + kBaseThreadNum-1) / kBaseThreadNum;
            dim3 dimBlock(kBaseThreadNum, 1, 1);

            if (num_block < kMaxGridNum) {
                dim3 dimGrid(num_block, 1, 1);
                MapPlanKernel<Saver, expr::Plan<E>, kBaseThreadBits>   \
                    <<<dimGrid,dimBlock>>>(dst, xstride, plan);
            } else {
                int repeat = (num_block + kBaseGridNum-1) / kBaseGridNum;
                dim3 dimGrid( kBaseGridNum, 1 , 1 );
                MapPlanLargeKernel<Saver,expr::Plan<E>, kBaseThreadBits, kBaseGridNum> \
                    <<<dimGrid,dimBlock>>>(dst, xstride, plan, repeat );
            }
        }        
    }; // namespace cuda
    
    namespace cuda{
        template<typename Saver,typename Reducer, int warp_bits, typename Plan>
        __global__ void MapRedKeepLowestKernel( Tensor<gpu,1> dst, Plan plan, real_t scale, Shape<2> eshape ){
            const unsigned warp_size = 1 << warp_bits;
            const unsigned x = (blockIdx.x<<warp_bits) + threadIdx.x;
            // to avoid bank conflict
            __shared__ real_t s_res[ warp_size ][ warp_size + 1 ];

            // note: reverse store [y][x], so that we can reduce over threadIdx.x, use warp optimization
            if( threadIdx.y < eshape[1] && x < eshape[0] ){
                s_res[ threadIdx.x ][ threadIdx.y ] = plan.Eval( threadIdx.y, x );
            }
            for( unsigned y = warp_size; y < eshape[1]; y += warp_size ){
                if( threadIdx.y + y < eshape[1] && x < eshape[0] ){
                    Reducer::Reduce( s_res[ threadIdx.x ][ threadIdx.y ], plan.Eval( threadIdx.y + y, x ) );
                }
            } 
            __syncthreads();
            if( eshape[1] >= warp_size ){
                Reduce1D<Reducer,warp_bits>( s_res[ threadIdx.y ] );
            }else{
                Reduce1DNotAlign<Reducer,warp_bits>( s_res[ threadIdx.y ], eshape[1] );
            }
            __syncthreads();            
            
            if( threadIdx.y == 0 && x < eshape[0] ){
                Saver::Save( dst[x],  s_res[ threadIdx.x ][ 0 ] * scale );
            } 
        }        
        
        template<typename Saver, typename Reducer, typename E>
        inline void MapReduceKeepLowest( Tensor<gpu,1> dst, const expr::Plan<E> &plan, real_t scale, Shape<2> eshape ){
            dim3 dimBlock( kMemUnit, kMemUnit );
            dim3 dimGrid ( (eshape[0]+kMemUnit-1) >> kMemUnitBits );
            CheckLaunchParam( dimGrid, dimBlock, "MapRedKeepLowestKernel" );
            MapRedKeepLowestKernel<Saver,Reducer,kMemUnitBits><<<dimGrid,dimBlock>>>( dst, plan, scale, eshape );
        } 
    }; // namespace cuda
    
    namespace cuda{
        template<typename Saver,typename Reducer, int block_dim_bits, typename Plan>
        __global__ void MapReduceKeepDim2Kernel( Tensor<gpu,1> dst, Plan plan, real_t scale, Shape<4> pshape ){
            const int block_size = 1 << block_dim_bits;
            __shared__ real_t s_rec[ block_size ];
            const int c = blockIdx.x;            
            const index_t tot = pshape[0]*pshape[1]*pshape[3];

            real_t res = Reducer::kInitV;
            for( index_t i_offset = 0; i_offset < tot; i_offset += block_size ){
                index_t i = i_offset + threadIdx.x;
                if( i< tot ){
                    const index_t x = i % pshape[0];
                    i /= pshape[0]; 
                    const index_t y = i % pshape[1];
                    const index_t n = i / pshape[1];
                    Reducer::Reduce( res, plan.Eval( (n*pshape[2] + c) * pshape[1] + y, x ) );
                }
            }                
            s_rec[ threadIdx.x ] = res;
            __syncthreads();
            Reduce1D<Reducer,block_dim_bits>( s_rec );
            if( threadIdx.x == 0 ){
                Saver::Save( dst[c], s_rec[0]*scale );
            }
        }

        template<typename Saver, typename Reducer, typename Plan>
        inline void MapReduceKeepDim2( Tensor<gpu,1> dst, const Plan &plan, real_t scale, Shape<4> pshape ){  
            dim3 dimBlock( kBaseThreadNum );
            dim3 dimGrid ( dst.shape[0] );
            CheckLaunchParam( dimGrid, dimBlock, "MapReduceKeepDim2" );
            MapReduceKeepDim2Kernel<Saver,Reducer,kBaseThreadBits>
                <<<dimGrid,dimBlock>>>( dst, plan, scale, pshape );
        }
    };
    
    namespace cuda{
        template<int x_bits>        
        __global__ void SoftmaxKernel( Tensor<gpu,2> dst, Tensor<gpu,2> src ){
            const unsigned x_size = 1 << x_bits;  
            const int y = blockIdx.x;
            __shared__ real_t s_rec[ x_size ];
            
            // step 1: get max
            if( threadIdx.x < dst.shape[ 0 ] ){
                s_rec[ threadIdx.x ] = src[ y ][ threadIdx.x ] ; 
            }
            for( unsigned x = x_size; x < dst.shape[0]; x += x_size ){
                if( x + threadIdx.x < dst.shape[0] ){
                    real_t a = src[ y ][ x + threadIdx.x ];
                    s_rec[ threadIdx.x ] = max( a, s_rec[ threadIdx.x] );
                }
            }
            __syncthreads();
            if( threadIdx.x >= dst.shape[0] ){
                s_rec[ threadIdx.x ] = s_rec[0];
            }
            __syncthreads();
            Reduce1D<red::maximum,x_bits>( s_rec );
            __syncthreads();
            real_t smax = s_rec[0];            
            __syncthreads();
            s_rec[ threadIdx.x ] = 0.0f;
            __syncthreads();

            // calculate normalizer, with writeback
            for( unsigned x = 0; x < dst.shape[0]; x += x_size ){
                if( x + threadIdx.x < dst.shape[0] ){
                    real_t p = expf( src[ y ][ x + threadIdx.x ] - smax );
                    s_rec[ threadIdx.x ] += p;
                    // write back first, will fetch later
                    dst[ y ][ x + threadIdx.x ] = p;
                }
            }
            // calculate normalizer
            __syncthreads();
            Reduce1D<red::sum,x_bits>( s_rec );
            __syncthreads();
            real_t ssum = s_rec[0];

            for( unsigned x = 0; x < dst.shape[0]; x += x_size ){
                if( x + threadIdx.x < dst.shape[0] ){
                    dst[ y ][ x + threadIdx.x ] /= ssum;
                }
            }
        }
    
        inline void Softmax( Tensor<gpu,2> &dst, const Tensor<gpu,2> &src ){
            dim3 dimBlock( kBaseThreadNum );
            dim3 dimGrid ( dst.shape[1] );
            utils::Assert( dst.shape == src.shape, "Softmax: shape mismatch" );
            CheckLaunchParam( dimGrid, dimBlock, "Softmax" );
            SoftmaxKernel<kBaseThreadBits><<<dimGrid,dimBlock>>>( dst, src );
        }
    }; // namespace cuda
}; // namespace mshadow
#endif // TENSOR_GPU_INL_H

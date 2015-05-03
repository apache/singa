#ifndef MSHADOW_TENSOR_IO_H
#define MSHADOW_TENSOR_IO_H
/*!
 * \file tensor_io.h
 * \brief definitions of I/O functions for mshadow tensor
 * \author Tianqi Chen
 */
#include <cstdio>
#include "tensor.h"

namespace mshadow{
    namespace utils{
        /*! 
         * \brief interface of stream I/O, used to serialize data, 
         *   it is not restricted to only this interface in SaveBinary/LoadBinary
         *   mshadow accept all class that implements Read and Write
         */
        class IStream{
        public:
            /*! 
             * \brief read data from stream
             * \param ptr pointer to memory buffer
             * \param size size of block
             * \return usually is the size of data readed
             */
            virtual size_t Read( void *ptr, size_t size ) = 0;        
            /*! 
             * \brief write data to stream
             * \param ptr pointer to memory buffer
             * \param size size of block
             */
            virtual void Write( const void *ptr, size_t size ) = 0;
            /*! \brief virtual destructor */
            virtual ~IStream( void ){}
        };
    };
    
    /*! 
     * \brief CPU/GPU: save a tensor by binary format, for GPU version, a temp Tensor<cpu,dim> storage will be allocated
     * \param fo output binary stream
     * \param src source data file
     * \tparam dim dimension of tensor
     * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
     */
    template<int dim,typename TStream>
    inline void SaveBinary( TStream &fo, const Tensor<cpu,dim> &src );
    /*! \brief refer to comment of cpu ver \sa SaveBinary */
    template<int dim,typename TStream>
    inline void SaveBinary( TStream &fo, const Tensor<gpu,dim> &src );

    /*! 
     * \brief CPU/GPU: load a tensor by binary format, for GPU version, a temp Tensor<cpu,dim> storage will be allocated
     *       if pre_alloc is true , then space in dst is preallocated, and must have same shape of the tensor loaded
     *       if pre_alloc is false, then dst originally does not have space allocated, LoadBinary will allocate space for dst
     * \param fi output binary stream
     * \param dst destination file
     * \param pre_alloc whether space is pre-allocated, if false, space allocation will happen
     * \tparam dim dimension of tensor     
     * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
     */
    template<int dim,typename TStream>
    inline void LoadBinary( TStream &fi, Tensor<cpu,dim> &dst, bool pre_alloc );
    /*! \brief refer to comment of cpu ver \sa LoadBinary */
    template<int dim,typename TStream>
    inline void LoadBinary( TStream &fi, Tensor<gpu,dim> &dst, bool pre_alloc );
    
    namespace utils{
        /*! \brief implementation of file i/o stream */
        class FileStream: public IStream{
        public:
            /*! \brief constructor */
            FileStream( FILE *fp ):fp_(fp){}
            virtual size_t Read( void *ptr, size_t size ){
                return fread( ptr, size, 1, fp_ );
            }
            virtual void Write( const void *ptr, size_t size ){
                fwrite( ptr, size, 1, fp_ );
            }
            /*! \brief close file */
            inline void Close( void ){
                fclose( fp_ );
            }
        private:
            FILE *fp_;
        };
    };
};

namespace mshadow{
    // implementations
    template<int dim, typename TStream>
    inline void SaveBinary( TStream &fo, const Tensor<cpu,dim> &src_ ){
        fo.Write( src_.shape.shape_, sizeof(index_t) * dim );
        Tensor<cpu,2> src = src_.FlatTo2D();
        for( index_t i = 0; i < src.shape[1]; ++ i ){
            fo.Write( src[i].dptr, sizeof(real_t)*src.shape[0] );
        }
    }
    template<int dim, typename TStream>
    inline void SaveBinary( TStream &fo, const Tensor<gpu,dim> &src ){
        // copy to CPU, then save
        Tensor<cpu,dim> tmp( src.shape ); 
        AllocSpace( tmp );
        Copy( tmp, src );
        SaveBinary( fo, tmp );
        FreeSpace( tmp );
    }

    template<int dim, typename TStream>
    inline void LoadBinary( TStream &fi, Tensor<cpu,dim> &dst_, bool pre_alloc ){
        Shape<dim> shape;
        utils::Assert( fi.Read( shape.shape_, sizeof(index_t) * dim ) != 0, "mshadow::LoadBinary" );
        if( pre_alloc ){
            utils::Assert( shape == dst_.shape );
        }else{
            dst_.shape = shape; AllocSpace( dst_ );
        }
        Tensor<cpu,2> dst = dst_.FlatTo2D();
        if( dst.shape[0] == 0 ) return;        
        for( index_t i = 0; i < dst.shape[1]; ++ i ){
            utils::Assert( fi.Read( dst[i].dptr, sizeof(real_t)*dst.shape[0] ) != 0, "mshadow::LoadBinary" );
        }
    } 
    template<int dim, typename TStream>
    inline void LoadBinary( TStream &fi, Tensor<gpu,dim> &dst, bool pre_alloc ){
        Tensor<cpu,dim> tmp;
        LoadBinary( fi, tmp, false );
        if( pre_alloc ){
            utils::Assert( tmp.shape == dst.shape );
        }else{
            dst.shape = tmp.shape; AllocSpace( dst );
        }
        Copy( dst, tmp );
        FreeSpace( tmp );
    }
};
#endif // TENSOR_IO_H

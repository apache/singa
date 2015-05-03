#ifndef MSHADOW_TENSOR_CONTAINER_H
#define MSHADOW_TENSOR_CONTAINER_H
/*!
 * \file tensor_container.h
 * \brief tensor container that does memory allocation and resize like STL
 * \author Tianqi Chen
 */
#include "tensor.h"
#include "tensor_io.h"

namespace mshadow{
    /*!
     * \brief tensor container that does memory allocation and resize like STL,
     *        use it to save the lines of FreeSpace in class.
     *        Do not abuse it, efficiency can come from pre-allocation and no re-allocation
     *
     * \tparam Device which device the tensor is on
     * \tparam dimension dimension of the tensor
     */
    template<typename Device, int dimension>
    class TensorContainer: public Tensor<Device,dimension>{
    public:
        /*! 
         * \brief constructor 
         * \param pad whether use padding alignment in space allocation
         */
        TensorContainer( bool pad = MSHADOW_ALLOC_PAD ){
            this->pad_ = pad;
            this->dptr = data_.dptr = NULL;
            this->shape[0] = 0;
            this->shape.stride_ = 0;
            this->data_.shape.stride_ = 0;
            this->data_.shape[1] = 0;
        }
        /*! 
         * \brief constructor 
         * \param shape intial shape
         */
        TensorContainer( const Shape<dimension> &shape ){
            this->pad_ = MSHADOW_ALLOC_PAD;
            data_.dptr = NULL;
            this->AllocByShape( shape );
        }
        /*! 
         * \brief constructor 
         * \param shape intial shape
         * \param initv intial value
         */
        TensorContainer( const Shape<dimension> &shape, real_t initv ){
            this->pad_ = MSHADOW_ALLOC_PAD;
            data_.dptr = NULL;
            this->AllocByShape( shape );
            (*this) = initv;
        }
        ~TensorContainer( void ){
            this->FreeSpace();
        }
        /*! 
         * \brief resize the container to given shape, content is NOT preserved
         * \param shape target shape
         */
        inline void Resize( const Shape<dimension> &shape ){
            Shape<2> s2 = shape.FlatTo2D();            
            if( s2.shape_[0] > data_.shape.stride_ || s2.shape_[1] > data_.shape[1] ){
                this->AllocByShape( shape );
            }else{
                this->shape = shape;
                if( this->pad_ ){
                    this->shape.stride_ = data_.shape.stride_;
                }else{
                    this->shape.stride_ = this->shape[ 0 ];
                }
            }
        }
        /*! 
         * \brief resize the container to given shape, and initialize, content is NOT preserved
         * \param shape target shape
         * \param initv initialization value
         */
        inline void Resize( const Shape<dimension> &shape, real_t initv ){
            this->Resize( shape );
            (*this) = initv;
        }
        /*! \brief set whether padding is allowed in tensor */
        inline void set_pad( bool pad ){
            this->pad_ = pad;
        }
        /*! 
         * \brief save by binary format
         * \param fo output binary stream
         * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
         */
        template<typename TStream>
        inline void SaveBinary( TStream &fo ) const{
            mshadow::SaveBinary( fo, *this );
        }
        /*! 
         * \brief load by binary format, a temp Tensor<cpu,dim> storage will be allocated
         * \param fi input binary stream
         * \tparam TStream type of stream, need to support Read, Write, one example is utils::IStream.
         */
        template<typename TStream>
        inline void LoadBinary( TStream &fi ) {
            Tensor<cpu,dimension> tmp;
            mshadow::LoadBinary( fi, tmp, false );
            this->Resize( tmp.shape );
            Copy( *this, tmp );
            mshadow::FreeSpace( tmp );
        }
    public:
        // functions to fit exp template
        inline Tensor<Device,dimension>& operator=( real_t s ){
            return this->__assign( s );
        }
        template<typename E>
        inline Tensor<Device,dimension>& operator=( const expr::Exp<E,expr::type::kMapper> &exp ){
            return this->__assign( exp );
        }
        template<typename E>
        inline Tensor<Device,dimension>& operator=( const expr::Exp<E,expr::type::kComplex> &exp ){
            return this->__assign( exp );
        }
    private:
        /*! \brief whether we do padding in the space */
        bool pad_;
        /*! \brief the shape of data_ is actually current data space */
        Tensor<Device, 2> data_;
    private:
        inline void FreeSpace (void){
            if( data_.dptr != NULL ){
                mshadow::FreeSpace( data_ );
                data_.dptr = this->dptr = NULL;
            }
        }
        inline void AllocByShape (const Shape<dimension>& shape){
            if( data_.dptr != NULL ){
                this->FreeSpace();
            }
            data_.shape = shape.FlatTo2D();
            mshadow::AllocSpace( data_, pad_ );
            this->dptr  = data_.dptr;
            this->shape = shape;
            if( this->pad_ ){
                this->shape.stride_ = data_.shape.stride_;
            }else{
                this->shape.stride_ = shape[0];
            }
        }
    };
};// namespace mshadow

#endif

#ifndef MSHADOW_TENSOR_H
#define MSHADOW_TENSOR_H
/*!
 * \file tensor.h
 * \brief header file of tensor data structure and functions
 *        covention: this lib requires explicit memory allocation and de-allocation
 *                   all the data structure Tensor<cpu,1>, Tensor<gpu,1> are like handles(pointers),
 *                   no memory allocation is happening during calculation
 * \author Bing Xu, Tianqi Chen
 */
#include "tensor_base.h"
#include "tensor_expr.h"

namespace mshadow {
    /*!
     * \brief shape of a tensor
     *       IMPORTANT NOTE: this shape is different from numpy.shape
     *       shape[0] gives the lowest dimension, shape[dimension-1] gives the highest dimension
     *       shape[k] corresponds to k-th dimension of tensor
     * \tparam dimension dimension of tensor
     */
    template<int dimension>
    struct Shape {
    public:
        /*! \brief maximum dimension of tensor */
        const static int kMaxShape = dimension;
        /*! \brief maximum dimension minus 1 */
        const static int kSubShape = dimension - 1;
    public:
        /*! \brief default constructor, do nothing */
        MSHADOW_XINLINE Shape(void) {}
        /*! \brief constuctor */
        MSHADOW_XINLINE Shape( const Shape<dimension> &s ){
            #pragma unroll
            for( int i = 0; i < kMaxShape; ++i ){
                this->shape_[i] = s[i];
            }
            this->stride_ = s.stride_;
        }
        /*!
         * \brief get corresponding index
         * \param idx dimension index
         * \return the corresponding dimension size
         */
        MSHADOW_XINLINE index_t& operator[](index_t idx) {
            return shape_[ idx ];
        }
        /*!
         * \brief get corresponding index
         * \param idx dimension index
         * \return the corresponding dimension size
         */
        MSHADOW_XINLINE const index_t& operator[](index_t idx) const {
            return shape_[ idx ];
        }
        /*! \return whether two shape equals */
        MSHADOW_XINLINE bool operator==(const Shape<kMaxShape> &s) const {
            #pragma unroll
            for ( int i = 0; i < kMaxShape; ++i ) {
                if (s.shape_[i] != this->shape_[i]) return false;
            }
            return true;
        }
        /*!
         * flatten the higher dimension to second dimension, return a 2D shape
         * \return the flat 2d shape
         */
        MSHADOW_XINLINE Shape<2> FlatTo2D(void) const {
            Shape<2> s;
            s.stride_ = this->stride_;
            s.shape_[ 0 ] = this->shape_[ 0 ];
            index_t ymax = 1;

            #pragma unroll
            for (int i = 1; i < kMaxShape; ++i) {
                ymax *= this->shape_[ i ];
            }
            s.shape_[1] = ymax;
            return s;
        }
        /*! \return number of valid elements */
        MSHADOW_XINLINE size_t Size(void) const{
            size_t memsz = this->shape_[ 0 ];
            #pragma unroll
            for (int i = 1; i < kMaxShape; ++i) {
                memsz *= this->shape_[ i ];
            }
            return memsz;
        }
        /*! \return memory size, including the aligned x dimension */
        MSHADOW_XINLINE size_t MSize(void) const {
            size_t memsz = this->stride_;
            #pragma unroll
            for (int i = 1; i < kMaxShape; ++i) {
                memsz *= this->shape_[ i ];
            }
            return memsz;
        }
        /*!
         * \return product shape in [dimstart,dimend)
         * \param dimstart start dimension
         * \param dimend   end dimension
         */
        MSHADOW_XINLINE index_t ProdShape( int dimstart, int dimend ) const{
            index_t num = 1;
            #pragma unroll
            for (int i = dimstart; i < dimend; ++i) {
                num *= this->shape_[ i ];
            }
            return num;
        }
        /*!
         * \brief get subshape
         * \return subshape
         */
        MSHADOW_XINLINE Shape<kSubShape> SubShape(void) const {
            Shape<kSubShape> s;
            s.stride_ = this->stride_;
            // for cuda
            #pragma unroll
            for (int i = 0; i < kSubShape; ++i) {
                s.shape_[ i ] = this->shape_[ i ];
            }
            return s;
        }

    public:
        /*! \brief storing the dimension information */
        index_t shape_[ kMaxShape ];
        /*!
         * \brief storing the stride information in x dimension
         *    this is used to deal with pitch allocation in gpu or sse(align x dimension to 64bit) for efficiency
         */
        index_t stride_;
    };
    // useful construction functions to generate shape
    /*!
     * \brief construct a one dimension shape, stride will equal s0
     * \param s0 size of dimension 0
     * \return the shape construction
     */
    MSHADOW_XINLINE Shape<1> Shape1( index_t s0 ){
        Shape<1> s; s[0] = s0; s.stride_ = s0;
        return s;
    }
    /*!
     * \brief construct a two dimension shape, stride will equal s0
     * \param s1 size of dimension 1
     * \param s0 size of dimension 0
     * \return the shape construction
     */
    MSHADOW_XINLINE Shape<2> Shape2( index_t s1, index_t s0 ){
        Shape<2> s; s[0] = s0; s[1] = s1; s.stride_ = s0;
        return s;
    }
    /*!
     * \brief construct a three dimension shape, stride will equal s0
     * \param s2 size of dimension 2
     * \param s1 size of dimension 1
     * \param s0 size of dimension 0
     * \return the shape construction
     */
    MSHADOW_XINLINE Shape<3> Shape3( index_t s2, index_t s1, index_t s0 ){
        Shape<3> s;
        s[0] = s0; s[1] = s1; s[2] = s2; s.stride_ = s0;
        return s;
    }
    /*!
     * \brief construct a four dimension shape, stride will equal s0
     * \param s3 size of dimension 3
     * \param s2 size of dimension 2
     * \param s1 size of dimension 1
     * \param s0 size of dimension 0
     * \return the shape construction
     */
    MSHADOW_XINLINE Shape<4> Shape4( index_t s3, index_t s2, index_t s1, index_t s0 ){
        Shape<4> s;
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3; s.stride_ = s0;
        return s;
    }
}; // namespace mshadow

namespace mshadow {
    /*! \brief device name CPU */
    struct cpu {
        /*! \brief whether this device is CPU or not */
        const static bool kDevCPU = true;
        /*! \brief device flag number, identifies this device */
        const static int kDevMask = 1<<0;
    };
    /*! \brief device name CPU */
    struct gpu {
        /*! \brief whether this device is CPU or not */
        const static bool kDevCPU = false;
        /*! \brief device flag number, identifies this device */
        const static int kDevMask = 1<<1;
    };

    // more compact template
    /*!
     * \brief general tensor
     * \tparam Device which device the tensor is on
     * \tparam dimension dimension of the tensor
     */
    template<typename Device, int dimension>
    struct Tensor: public expr::ContainerExp< Tensor<Device,dimension> >{
    public:
        /*! \brief whether current type lies in cpu */
        const static bool kDevCPU = Device::kDevCPU;
        /*! \brief dimension of subtype */
        const static int  kSubdim = dimension - 1;

    public:
        /*! \brief pointer to the data */
        real_t *dptr;
        /*! \brief shape of the tensor */
        Shape<dimension> shape;
    public:
        /*! \brief default constructor */
        MSHADOW_XINLINE Tensor(void) {}
        /*! \brief constructor from shape  */
        MSHADOW_XINLINE Tensor(const Shape<dimension> &shape): shape(shape) {}
        /*! \brief constructor from data pointer and shape  */
        MSHADOW_XINLINE Tensor(real_t *dptr, const Shape<dimension> &shape): dptr((real_t*)dptr), shape(shape) {}
        /*!
         * \brief flatten the tensor to 2 dimension, collapse the higher dimensions together
         * \return tensor after flatten
         */
        MSHADOW_XINLINE Tensor<Device, 2> FlatTo2D(void) const {
            return Tensor<Device, 2>(reinterpret_cast<real_t*> \
                                     (dptr), shape.FlatTo2D());
        }
        /*!
         * \brief get a element of dimension - 1
         * \param idx index
         * \return the result tensor
         */
        MSHADOW_XINLINE Tensor<Device, kSubdim> operator[](index_t idx) const {
            Shape<kSubdim> s = shape.SubShape();
            return Tensor<Device, kSubdim>(reinterpret_cast<real_t*> \
                                           (dptr) + s.MSize() * idx, s);
        }
        /*!
         * \brief slice the tensor in highest dimension [begin,end)
         * \param begin begin position of slice
         * \param end end position of slice
         * \return tensor after slice
         */
        MSHADOW_XINLINE Tensor<Device, dimension> Slice(index_t begin, index_t end) const {
            Shape<dimension> s = this->shape;
            s[ dimension - 1 ] = end - begin;
            return Tensor<Device, dimension>(reinterpret_cast<real_t*>\
                                             (dptr) + s.SubShape().MSize() * begin, s);
        }
    public:
        /*!\brief functions to fit expression template */
        inline Tensor<Device,dimension>& operator=( real_t s ){
            return this->__assign( s );
        }
        /*!\brief functions to fit expression template */
        template<typename E>
        inline Tensor<Device,dimension>& operator=( const expr::Exp<E,expr::type::kMapper> &exp ){
            return this->__assign( exp );
        }
        /*!\brief functions to fit expression template */
        template<typename E>
        inline Tensor<Device,dimension>& operator=( const expr::Exp<E,expr::type::kComplex> &exp ){
            return this->__assign( exp );
        }
    };

    /*
     *  respecialized class Tensor1D,thei is due to different implementation in operator[]
     */
    template<typename Device>
    struct Tensor<Device,1>: public expr::ContainerExp< Tensor<Device,1> >{
    public:
        real_t *dptr;
        Shape<1> shape;
    public:
        MSHADOW_XINLINE Tensor(void) {}
        MSHADOW_XINLINE Tensor(const Shape<1> &shape): shape(shape) {}
        MSHADOW_XINLINE Tensor(real_t *dptr, Shape<1> shape) :dptr(dptr), shape(shape) {}

        MSHADOW_XINLINE Tensor<Device, 2> FlatTo2D(void) const {
            return Tensor<Device, 2>(reinterpret_cast<real_t*> \
                                     (dptr), shape.FlatTo2D());
        }
        MSHADOW_XINLINE Tensor<Device, 1> Slice(index_t begin, index_t end) const {
            Shape<1> s;
            s[0] = s.stride_ = end  - begin;
            return Tensor<Device, 1>(reinterpret_cast<real_t*> \
                                     (dptr) + begin, s);
        }
        MSHADOW_XINLINE real_t &operator[](index_t idx) { return dptr[ idx ]; }
        MSHADOW_XINLINE const real_t &operator[](index_t idx)const { return dptr[ idx ]; }
    public:
        // functions to fit expression template
        inline Tensor<Device,1>& operator=( double s ){
            return this->__assign( s );
        }
        template<typename E>
        inline Tensor<Device,1>& operator=( const expr::Exp<E,expr::type::kMapper> &exp ){
            return this->__assign( exp );
        }
        template<typename E>
        inline Tensor<Device,1>& operator=( const expr::Exp<E,expr::type::kComplex> &exp ){
            return this->__assign( exp );
        }
    };
}; // namespace mshadow

// add unroll loops for the shape
namespace mshadow {
    // function declarations
    /*!
     * \brief initialize tensor engine, used to call intialization functions of dependent libs
     *        this function should be called before all GPU tensor operations,
     *        for using tensors in CPU, this call is actually not needed
     * \param device_id GPU device id to be choosed
     */
    inline void InitTensorEngine( int device_id=0 );
    /*!
     * \brief Shutdown tensor engine,
     *        this function should be called after all GPU tensor operations,
     *        for using tensors in CPU, this call is actually not needed
     */
    inline void ShutdownTensorEngine( void );

    /*!
     * \brief CPU/CPU: allocate space for CTensor, according to the shape in the obj
     *        this function is responsible to set the stride_ in each obj.shape
     * \tparam dim specify the dim of tensor
     * \param obj the tensor object, with shape specified
     * \param pad whether padding dimension 0, to make last dimension aligned,
     *            padding may help improve efficiency of matrix multiplications
     *            if true, will allocate space with stride_ that may not equals shape[0]
     *            if false, will allocate continuous space
     */
    template<int dim>
    inline void AllocSpace(Tensor<cpu,dim> &obj, bool pad = MSHADOW_ALLOC_PAD);
    /*! \brief refer to comment of cpu ver \sa AllocSpace */
    template<int dim>
    inline void AllocSpace(Tensor<gpu,dim> &obj, bool pad = MSHADOW_ALLOC_PAD);

    /*!
     * \brief CPU/GPU: free the space of tensor, will set obj.dptr to NULL
     * \tparam dim specify the dim of tensor
     * \param obj the tensor object
     */
    template<int dim>
    inline void FreeSpace(Tensor<cpu,dim> &obj);
    /*! \brief refer to comment of cpu ver \sa FreeSpace */
    template<int dim>
    inline void FreeSpace(Tensor<gpu,dim> &obj);

    /*!
     * \brief CPU/GPU: short cut to allocate and initialize a Tensor
     * \tparam Device device of tensor
     * \tparam dim dimention of tensor
     * \param shape: shape of tensor
     * \param initv: initialization value
     * \param pad : padding option
     * \sa AllocSpace
     */
    template<typename Device, int dim>
    inline Tensor<Device,dim> NewTensor(const Shape<dim> &shape, real_t initv, bool pad = MSHADOW_ALLOC_PAD);

    /*!
     * \brief copy data from one tensor to another, with same shape
     * \tparam dim specify the dim of tensor
     * \param dst target tensor
     * \param src source tensor
     */
    template<int dim>
    inline void Copy(Tensor<cpu,dim> dst, const Tensor<cpu,dim> &src );
    /*! \brief refer to comment of cpu ver \sa Copy */
    template<int dim>
    inline void Copy(Tensor<cpu,dim> dst, const Tensor<gpu,dim> &src );
    /*! \brief refer to comment of cpu ver \sa Copy */
    template<int dim>
    inline void Copy(Tensor<gpu,dim> dst, const Tensor<cpu,dim> &src );
    /*! \brief refer to comment of cpu ver \sa Copy */
    template<int dim>
    inline void Copy(Tensor<gpu,dim> dst, const Tensor<gpu,dim> &src );


    /*!
     * \brief CPU/GPU: normalize softmax: dst[i][j] = exp( energy[i][j] ) /( sum_j exp( energy[i][j] ) )
     * \param dst destination
     * \param energy input energy
     */
    inline void Softmax( Tensor<cpu,2> dst, const Tensor<cpu,2> &energy );
    /*! \brief refer to comment of cpu ver \sa Softmax */
    inline void Softmax( Tensor<gpu,2> dst, const Tensor<gpu,2> &energy );

}; // namespace mshadow


namespace mshadow{
    // function declarations to support expression, no need to understand them
    // these functions do not need to be directly used

    /*!
     * \brief CPU/GPU: map a expression to a tensor, this function calls MapPlan
     * \tparam Saver specify storage method
     * \tparam dim dim of the tensor, during usage, there is no need to specify this parameter
     * \tparam E specifies the expression type, not need to specify this parameter during usage
     * \tparam etype expression type
     * \param dst destination
     * \param exp expression
     * \sa namespace mshadow:sv, mshadow::op, mshadow::expr
     */
    template<typename Saver, int dim, typename E, int etype>
    inline void MapExp(Tensor<cpu,dim> dst, const expr::Exp<E,etype> &exp );
    /*! \brief refer to comment of cpu ver \sa MapExp */
    template<typename Saver, int dim, typename E, int etype>
    inline void MapExp(Tensor<gpu,dim> dst, const expr::Exp<E,etype> &exp );

    /*!
     * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in lowest dimension (dimension 0)
     * \tparam Saver specify storage method
     * \tparam Reducer specify a reducer method
     * \tparam E specifies the expression type, not need to specify this parameter during usage
     * \tparam etype expression type
     * \param dst destination
     * \param exp expression
     * \param scale scale the result before save
     * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
     */
    template<typename Saver, typename Reducer, typename E, int etype>
    inline void MapReduceKeepLowest( Tensor<cpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale = 1.0f );
    /*! \brief refer to comment of cpu ver \sa MapReduceKeepLowest */
    template<typename Saver, typename Reducer, typename E, int etype>
    inline void MapReduceKeepLowest( Tensor<gpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale = 1.0f );


    /*!
     * \brief CPU/GPU: map a expression, do reduction to 1D Tensor in third dimension (dimension 2)
     * \tparam Saver specify storage method
     * \tparam Reducer specify a reducer method
     * \tparam E specifies the expression type, not need to specify this parameter during usage
     * \tparam dimkeep the target dimension to be kept, should be larger than 0, for 0, use MapReduceKeepLowest
     * \tparam etype expression type
     * \param dst destination
     * \param exp expression
     * \param scale scale the result before save
     * \sa namespace mshadow:sv, mshadow::op, mshadow::red, mshadow::expr
     */
    template<typename Saver, typename Reducer, int dimkeep, typename E, int etype>
    inline void MapReduceKeepHighDim( Tensor<cpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale = 1.0f );
    /*! \brief refer to comment of cpu ver \sa MapReduceKeepHighDim */
    template<typename Saver, typename Reducer, int dimkeep, typename E, int etype>
    inline void MapReduceKeepHighDim( Tensor<gpu,1> dst, const expr::Exp<E,etype> &exp, real_t scale = 1.0f );

};// namespace mshadow

// execution implementation of expression evaluations
#include "tensor_expr_engine-inl.hpp"
// cpu implementation of functions
#include "tensor_cpu-inl.hpp"
// gpu implementation of functions
#include "tensor_gpu-inl.hpp"
// extension of expressions
#include "tensor_expr_ext.h"
// io 
#include "tensor_io.h"
// container
#include "tensor_container.h"
// random number generator
#include "tensor_random.h"
#endif // TENSOR_H

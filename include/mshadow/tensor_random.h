#ifndef MSHADOW_TENSOR_RANDOM_H
#define MSHADOW_TENSOR_RANDOM_H
/*!
 *  \file tensor_random.h
 *  \brief Random inline functions for tensor.
 *  \author Bing Xu, Tianqi Chen
 *   Based on curand|MKL|stdlib
 */
#include <cstdlib>
#include <random>
#include <chrono>
#include "tensor.h"
#include "tensor_container.h"

namespace mshadow {
    /*!
     * \brief random number generator
     * \tparam Device the device of random number generator
     *
     * Note: replaced rand (srand) with c++11's random functions.
     */
    template<typename Device>
    class Random {};

    /*! \brief CPU random number generator */
    template<>
    class Random<cpu> {
    public:
        /*!
         * \brief constructor of random engine using default seed
         */
        Random<cpu> (){
          // obtain a seed from the system clock:
          unsigned s= std::chrono::system_clock::now().time_since_epoch().count();
          Seed(s);
        }
        /*!
         * \brief constructor of random engine
         * \param seed random number seed
         */
        Random<cpu>( int seed ){
            #if MSHADOW_USE_MKL
            int status = vslNewStream(&vStream_, VSL_BRNG_MT19937, seed);
            utils::Assert( status == VSL_STATUS_OK, "MKL VSL Random engine failed to be initialized.\n" );
            #else
            //srand(seed);
            gen_.seed(seed);
            #endif
            buffer_.Resize( Shape1( kRandBufferSize ) );
        }
        ~Random<cpu>() {
            #if MSHADOW_USE_MKL
            vslDeleteStream(&vStream_);
            #endif
        }
        /*!
         * \brief seed random number generator using this seed
         * \param seed seed of prng
         */
        inline void Seed( int seed ){
            #if MSHADOW_USE_MKL
            int status = vslDeleteStream(&vStream_);
            utils::Assert(status == VSL_STATUS_OK);
            status = vslNewStream(&vStream_, VSL_BRNG_MT19937, seed);
            utils::Assert(status == VSL_STATUS_OK);
            #else
            // srand( seed );
            gen_.seed(seed);
            #endif
        }
        /*!
         * \brief generate binary data according to a probability matrix
         * \param dst destination
         * \param a lower bound of uniform
         * \param b upper bound of uniform
         * \tparam dim dimension of tensor
         */
        template<int dim>
        inline void SampleBinary( Tensor<cpu, dim> &dst) {
            real_t a=0.0f;
            real_t b=1.0f;
            Tensor<cpu, 2> mat = dst.FlatTo2D();
            std::uniform_real_distribution<real_t> distribution (a,b);
            for ( index_t i = 0; i < mat.shape[1]; ++i ) {
                #if MSHADOW_USE_MKL
                #if MSHADOW_SINGLE_PRECISION
                int status = vsRngUniform( 0, vStream_, mat.shape[0], mat[i].dptr, a, b );
                #else
                int status = vdRngUniform( 0, vStream_, mat.shape[0], mat[i].dptr, a, b );
                #endif
                utils::Assert(status == VSL_STATUS_OK, "Failed to generate random number by MKL.\n" );
                #else
                // use stdlib
                /*
                for ( index_t j = 0; j < mat.shape[0]; ++j ) {
                    mat[i][j] = this->RandNext()*(b-a) + a;
                }
                */
                for ( index_t j = 0; j < mat.shape[0]; ++j ) {
                    mat[i][j] = distribution(gen_) > mat[i][j] ? 0.0f: 1.0f;
                }
                #endif
            }
        }
        /*!
         * \brief generate data from uniform [a,b)
         * \param dst destination
         * \param a lower bound of uniform
         * \param b upper bound of uniform
         * \tparam dim dimension of tensor
         */
        template<int dim>
        inline void SampleUniform( Tensor<cpu, dim> &dst, real_t a=0.0f, real_t b=1.0f ) {
            Tensor<cpu, 2> mat = dst.FlatTo2D();
            std::uniform_real_distribution<real_t> distribution (a,b);
            for ( index_t i = 0; i < mat.shape[1]; ++i ) {
                #if MSHADOW_USE_MKL
                #if MSHADOW_SINGLE_PRECISION
                int status = vsRngUniform( 0, vStream_, mat.shape[0], mat[i].dptr, a, b );
                #else
                int status = vdRngUniform( 0, vStream_, mat.shape[0], mat[i].dptr, a, b );
                #endif
                utils::Assert(status == VSL_STATUS_OK, "Failed to generate random number by MKL.\n" );
                #else
                // use stdlib
                /*
                for ( index_t j = 0; j < mat.shape[0]; ++j ) {
                    mat[i][j] = this->RandNext()*(b-a) + a;
                }
                */
                for ( index_t j = 0; j < mat.shape[0]; ++j ) {
                    mat[i][j] = distribution(gen_);
                }
                #endif
            }
        }
        /*!
         * \brief generate data from standard gaussian
         * \param dst destination
         * \param mu mean variable
         * \param sigma standard deviation
         * \tparam dim dimension of tensor
         */
        template<int dim>
        inline void SampleGaussian( Tensor<cpu, dim> &dst, real_t mu = 0.0f, real_t sigma = 1.0f ) {
            if( sigma <= 0.0f ) {
                dst = mu; return;
            }
            Tensor<cpu, 2> mat = dst.FlatTo2D();
            std::normal_distribution<real_t> distribution (mu, sigma);
            for (index_t i = 0; i < mat.shape[1]; ++i) {
                #if MSHADOW_USE_MKL
                #if MSHADOW_SINGLE_PRECISION
                int status = vsRngGaussian( 0, vStream_, mat.shape[0], mat[i].dptr, mu, sigma );
                #else
                int status = vdRngGaussian( 0, vStream_, mat.shape[0], mat[i].dptr, mu, sigma );
                #endif
                utils::Assert(status == VSL_STATUS_OK, "Failed to generate random number by MKL.\n" );
                #else
                /*
                real_t g1 = 0.0f, g2 = 0.0f;
                for (index_t j = 0; j < mat.shape[0]; ++j) {
                    if( (j & 1) == 0 ){
                        this->SampleNormal2D( g1, g2 );
                        mat[i][j] = mu + g1 * sigma;
                    }else{
                        mat[i][j] = mu + g2 * sigma;
                    }
                }
                */
                for (index_t j = 0; j < mat.shape[0]; ++j) {
                  mat[i][j] = distribution(gen_);
                }
                #endif
            }
        }
        /*!
         * \brief return a temporal expression storing standard gaussian random variables
         *        the temporal tensor is only valid before next call of gaussian or uniform
         *        can be used as part of expression
         *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
         *           since second call of gaussian(s2) makes gaussian(s1) invalid
         *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
         * \param shape shape of the tensor
         * \tparam dim dimension of tensor
         */
        template<int dim>
        inline expr::ReshapeExp<Tensor<cpu,1>,dim,1> gaussian( Shape<dim> shape ){
            buffer_.Resize( Shape1( shape.Size() ) );
            this->SampleGaussian( buffer_, 0.0f, 1.0f );
            return expr::reshape( buffer_, shape );
        }
        /*!
         * \brief return a temporal expression storing standard uniform [0,1)
         *        the temporal tensor is only valid before next call of gaussian or uniform
         *        can be used as part of expression
         *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
         *           since second call of gaussian(s2) makes gaussian(s1) invalid
         *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
         * \param shape shape of the tensor
         * \tparam dim dimension of tensor
         */
        template<int dim>
        inline expr::ReshapeExp<Tensor<cpu,1>,dim,1> uniform( Shape<dim> shape ){
            buffer_.Resize( Shape1( shape.Size() ) );
            this->SampleUniform( buffer_, 0.0f, 1.0f );
            return expr::reshape( buffer_, shape );
        }
    private:
        /*! \brief get next random number from rand */
        inline real_t RandNext( void ){
            return static_cast<real_t>(rand()) / (static_cast<real_t>(RAND_MAX)+1.0f);
        }
        /*! \brief return a real numer uniform in (0,1) */
        inline real_t RandNext2( void ){
            return (static_cast<real_t>( rand() ) + 1.0 ) / (static_cast<real_t>(RAND_MAX) + 2.0);
        }
        /*!
         * \brief sample iid xx,yy ~N(0,1)
         * \param xx first  gaussian output
         * \param yy second gaussian output
         */
        inline void SampleNormal2D( real_t &xx, real_t &yy ){
            real_t x,y,s;
            do{
                x = 2.0f * RandNext2() - 1.0f;
                y = 2.0f * RandNext2() - 1.0f;
                s = x*x + y*y;
            }while( s >= 1.0f || s == 0.0f );
            real_t t = std::sqrt( -2.0f * std::log( s ) / s ) ;
            xx = x * t; yy = y * t;
        }
    private:
        #if MSHADOW_USE_MKL
        /*! \brief stream used by MKL VSL */
        VSLStreamStatePtr vStream_;
        #endif
        /*! \brief temporal space used to store random numbers */
        TensorContainer<cpu,1> buffer_;

        /*! \brief c++11 random generator, added for SINGA use */
        std::mt19937 gen_;
    }; // class Random<cpu>

#ifdef __CUDACC__

    /*! \brief GPU random number generator */
    template<>
    class Random<gpu> {
    public:
        /*!
         * \brief constructor of random engine
         * \param seed random number seed
         */
        Random<gpu>(int seed) {
            curandStatus_t status;
            status = curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
            utils::Assert(status == CURAND_STATUS_SUCCESS, "Can not create CURAND Generator");
            this->Seed( seed );
            buffer_.Resize( Shape1(kRandBufferSize) );
        }

        ~Random<gpu>() {
            curandStatus_t status;
            status = curandDestroyGenerator(gen_);
            utils::Assert(status == CURAND_STATUS_SUCCESS, "Destory CURAND Gen failed");
        }
        /*!
         * \brief seed random number generator using this seed
         * \param seed seed of prng
         */
        inline void Seed( int seed ){
            curandStatus_t status;
            status = curandSetPseudoRandomGeneratorSeed(gen_, seed);
            utils::Assert(status == CURAND_STATUS_SUCCESS, "Set CURAND seed failed.");
        }
        /*!
         * \brief generate data from uniform [a,b)
         * \param dst destination
         * \param a lower bound of uniform
         * \param b upper bound of uniform
         * \tparam dim dimension of tensor
         */
        template<int dim>
        inline void SampleUniform(Tensor<gpu, dim> &dst, real_t a=0.0f, real_t b=1.0f) {
            if( a == 0.0f && b == 1.0f ){
                dst = this->uniform( dst.shape );
            }else{
                dst = this->uniform( dst.shape ) *(b-a) + a;
            }
        }
        /*!
         * \brief generate data from standard gaussian
         * \param dst destination
         * \param mu mean variable
         * \param sigma standard deviation
         * \tparam dim dimension of tensor
         */
        template<int dim>
        inline void SampleGaussian(Tensor<gpu, dim> &dst, real_t mu = 0.0f, real_t sigma = 1.0f) {
            dst = this->gaussian( dst.shape, mu, sigma );
        }
        /*!
         * \brief return a temporal expression storing standard gaussian random variables
         *        the temporal tensor is only valid before next call of gaussian or uniform
         *        can be used as part of expression
         *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
         *           since second call of gaussian(s2) makes gaussian(s1) invalid
         *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
         * \param shape shape of the tensor
         * \param mu mean
         * \param sigma variance
         * \tparam dim dimension of tensor
         */
        template<int dim>
        inline expr::ReshapeExp<Tensor<gpu,1>,dim,1> gaussian( Shape<dim> shape, real_t mu=0.0f, real_t sigma=1.0f){
            size_t aligned_sz = ((shape.Size() + 1UL)>>1)<<1;
            // allocate alligned size
            buffer_.Resize( Shape1( aligned_sz ) );
            buffer_.Resize( Shape1( shape.Size() ) );
            curandStatus_t status;
            #if MSHADOW_SINGLE_PRECISION
            status = curandGenerateNormal(gen_, buffer_.dptr, aligned_sz , mu, sigma);
            #else
            status = curandGenerateNormalDouble(gen_, buffer_.dptr, buffer_.shape[0], mu, sigma);
            #endif
            utils::Assert(status == CURAND_STATUS_SUCCESS, "CURAND Gen Uniform failed\n");
            return expr::reshape( buffer_, shape );
        }
        /*!
         * \brief return a temporal expression storing standard uniform [0,1)
         *        the temporal tensor is only valid before next call of gaussian or uniform
         *        can be used as part of expression
         *  Caution: this means expression such as A = gaussian(s1) * gaussian(s2) will give invalid result,
         *           since second call of gaussian(s2) makes gaussian(s1) invalid
         *           A = gaussian(s1)*B+C; is correct; use one gaussian/uniform in each expression
         * \param shape shape of the tensor
         * \tparam dim dimension of tensor
         */
        template<int dim>
        inline expr::ReshapeExp<Tensor<gpu,1>,dim,1> uniform(Shape<dim> shape) {
            buffer_.Resize( Shape1( shape.Size() ) );
            curandStatus_t status;
            #if MSHADOW_SINGLE_PRECISION
            status = curandGenerateUniform(gen_, buffer_.dptr, buffer_.shape[0] );
            #else
            status = curandGenerateUniformDouble(gen_, buffer_.dptr, buffer_.shape[0] );
            #endif
            utils::Assert(status == CURAND_STATUS_SUCCESS, "CURAND Gen Uniform failed\n");
            return expr::reshape( buffer_, shape );
        }
    private:
        /*! \brief random numbeer generator */
        curandGenerator_t gen_;
        /*! \brief templ buffer */
        TensorContainer<gpu, 1> buffer_;
    }; // class Random<gpu>
    #endif

}; // namespace mshadow

#endif // MSHADOW_TENSOR_RANDOM_H

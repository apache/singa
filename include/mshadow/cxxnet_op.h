#ifndef CXXNET_OP_H
#define CXXNET_OP_H
#pragma once
/*!
 * \file cxxnet_op.h
 * \brief extra mshadow operation for cxxnet
 * \author Bing Xu
 */
#include "mshadow/tensor.h"

namespace mshadow {
    /*! \brief operations for algorithm */
    namespace op {
        struct sigmoid {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return 1.0f / (1.0f + expf(-a));
            }
        };
        struct sigmoid_grad {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return a * ( 1.0f - a );
            }
        };

        /*! \brief Rectified Linear Operation */
        struct relu {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                using namespace std;
                return max( a, 0.0f );
            }
        };
        struct relu_grad {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return a > 0.0f ? 1.0f : 0.0f;
            }
        };

        struct tanh {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return tanhf( a );
            }
        };
        struct tanh_grad {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return 1.0f - a * a;
            }
        };
        struct softplus {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return logf(1 + expf(a));
            }
        };
        struct softplus_grad {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return 1.0f / (1.0f + expf(-a));
            }
        };
        struct bnll {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return a > 0.0f ? a + logf(1.0f + expf(-a)) : logf(1.0f + expf(a));
            }
        };
        struct bnll_grad {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                real_t expval = a > 50.0f ? 50.0f : a; // kBNLL_THRESHOLD = 50.0f
                expval = expf(-expval);
                return 1.0f / (1.0f + expval);
            }
        };

        struct square {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return a * a;
            }
        };
       /*! \brief scaled tanh, hard code the scale factor*/
        struct stanh {
            MSHADOW_XINLINE static real_t Map(real_t a) {
              return  1.7159047*tanhf(0.66666667 *a);
            }
        };
        /*! \breif back prop for scaled tanh: */
        struct stanh_grad {
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return 0.66666667*1.7159047 -0.66666667/1.7159047*a*a;
            }
        };

        struct abs{
            MSHADOW_XINLINE static real_t Map(real_t a) {
                return a < 0 ? -a : a;
            }
        };

    }; //namespace op

}; //namespace mshadow

namespace mshadow {
    namespace op {
        /*! \brief used for generate Bernoulli mask */
        struct threshold {
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return a < b ? 1.0f : 0.0f;
            }
        };

        /*! \brief used for generate element of power */
        struct power {
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return powf( a, b );
            }
        };
        struct sqrtop {
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return sqrt(a+b);
            }
        };
        struct max {
            MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
                return a > b ? a : b;
            }
        };
    }; // namespace op
}; // namespace mshadow

#endif // CXXNET_OP_H

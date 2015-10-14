#ifndef SINGA_OP_H
#define SINGA_OP_H

#include<cmath>
#include <algorithm>

namespace op {
        struct Set {
            inline static void Map(float alpha, float & a) {
                a= alpha;
            }
        };

        struct Scale {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = alpha*a;
            }
        };
        struct Scale_grad {
            inline static void Map(float alpha,  float & output) {
                output = alpha;
            }
        };

        struct Exp {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = pow(a, alpha);
            }
        };
        struct Exp_grad {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = a * log(alpha);
                // log is the natrual log based on e
            }
        };

        struct Gsigmoid {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = 1.0f / (1.0f + expf(-a * alpha));
            }
        };
        struct Gsigmoid_grad {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = alpha * a * ( 1.0f - a );
            }
        };

        struct Grelu {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = ( 1 - alpha ) * std::max( a, 0.0f ) + alpha * a;
            }
        };
        struct Grelu_grad {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = a > 0.0f ? 1.0f : alpha;
            }
        };

        struct Gtanh {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = tanhf( a * alpha );
            }
        };
        struct Gtanh_grad {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = alpha * ( 1.0f - a * a );
            }
        };

        struct Softplus {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = logf(1 + expf(a));
            }
        };
        struct Softplus_grad {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = 1.0f / (1.0f + expf(-a));
            }
        };

        struct Square {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = a * a;
            }
        };

        struct Square_grad {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = 2 * sqrt(a);
            }
        };

        struct Sqrt {
            inline static void Map(float alpha,  const float & a, float & b) {
                b = sqrt(a);
            }
        };

        struct Threshold {
            inline static void Map(float alpha, float beta, const float & a, const float & b, float & c) {
                c =  a < b ? 1.0f : 0.0f;
            }
        };

        struct Add {
            inline static void Map(float alpha, float beta, const float & a, const float & b, float & c) {
                c =  a + b;
            }
        };

        struct Sub {
            inline static void Map(float alpha, float beta, const float & a, const float & b, float & c) {
                c =  a - b;
            }
        };

        struct Mult {
            inline static void Map(float alpha, float beta, const float & a, const float & b, float & c) {
                c =  a * b;
            }
        };

        struct Div {
            inline static void Map(float alpha, float beta, const float & a, const float & b, float & c) {
                c =  a / b;
            }
        };

        struct Sum {
            inline static void Map(const float * a, int n, float & b) {
                b = 0;
                for(int i = 0 ; i < n ; i++)
                {
                            b += a[i];
                }
            }
            };

        struct Repmat {
            inline static void Map(const float & a, int n, float * b) {
                for(int i = 0 ; i < n ; i++)
                {
                            b[i] = a;
                }
            }
        };

}; // namespace op

#endif // SINGA_OP_H

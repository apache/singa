/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/
#ifndef SINGA_UTILS_MATH_KERNEL_H_
#define SINGA_UTILS_MATH_KERNEL_H_

namespace singa {

extern "C" {
  void singa_gpu_softmaxloss_forward(int n, int dim, const float *prob,
      const int *label, float *loss);

  void singa_gpu_softmaxloss_backward(int n, int dim, float scale,
      const int *label, float *grad);

  void singa_gpu_sum_vec(float *data, float *sum , int n);

  void singa_gpu_sum_col(const float *src_mat_data, float *dst_vec_data,
    int rows, int cols, int stride);

  void singa_gpu_sum_row(const float *src_mat_data, float *dst_vec_data,
    int rows, int cols, int stride);

  void singa_gpu_add_vec_row(const float *src_vec_data,
    const float *src_mat_data, float *des_mat_data,
    int rows, int cols, int stride);

  void singa_gpu_exp(const float *src_data, float *des_data, int n);

  void singa_gpu_log(const float *src_data, float *des_data, int n);

  void singa_gpu_sigmoid(const float *src_data, float *des_data, int n);

  void singa_gpu_sigmoid_grad(const float *src_data, float *des_data, int n);

  void singa_gpu_relu(const float *src_data, float *des_data, int n);

  void singa_gpu_relu_grad(const float *src_data, float *des_data, int n);

  void singa_gpu_tanh(const float *src_data, float *des_data, int n);

  void singa_gpu_tanh_grad(const float *src_data, float *des_data, int n);

  void singa_gpu_softplus(const float *src_data, float *des_data, int n);

  void singa_gpu_softplus_grad(const float *src_data, float *des_data, int n);

  void singa_gpu_square(const float *src_data, float *des_data, int n);

  void singa_gpu_square_grad(const float *src_data, float *des_data, int n);

  void singa_gpu_sqrt(const float *src_data, float *des_data, int n);

  void singa_gpu_pow(const float *src_data_a, const float *src_data_b,
    float *des_data, int n);

  void singa_gpu_mult(const float *src_data_a, const float *src_data_b,
    float *des_data, int n);

  void singa_gpu_div(const float *src_data_a, const float *src_data_b,
    float *des_data, int n);

  void singa_gpu_set_value(float *data, float value, int n);

  void singa_gpu_threshold(const float *src_data, float *des_data,
      float alpha, int n);
};

}  // namespace singa

#endif  // SINGA_UTILS_MATH_KERNEL_H_

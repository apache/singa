/*********************************************************
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
 ************************************************************/
#include "batchnorm.h"

#include <cctype>

namespace singa {

BatchNormHandle::BatchNormHandle(const float momentum, const Tensor& input) {
  factor = momentum;
  batchsize = input.shape(0);
  channels = input.shape(1);
  if (input.nDim() == 4u) {
    height = input.shape().at(2);
    width = input.shape().at(3);
    is_2d = false;
  } else if (input.nDim() == 2u) {
    height = 1;
    width = 1;
    is_2d = true;
  } else {
    LOG(FATAL) << "The dimension of input should either be 4D or 2D.";
  }

#ifdef USE_DNNL
  if (input.device()->lang() == kCpp) {
    use_dnnl = true;
    epsilon = 1e-5f;
    x_dims = dnnl::memory::dims(input.shape().begin(), input.shape().end());

    // support f32 only
    auto dtype_ = memory::data_type::f32;
    memory::format_tag format_tag_ = get_dnnl_format_tag(input);
    x_md = dnnl::memory::desc({x_dims}, dtype_, format_tag_);

    // add to
    bn_fwd_training_d = new dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_training, x_md, epsilon,
        dnnl::normalization_flags::use_scale_shift);

    auto eng = input.device()->context(0)->dnnl_engine;
    bn_fwd_training_pd = new dnnl::batch_normalization_forward::primitive_desc(
        *bn_fwd_training_d, eng);
  }
#endif  // USE_DNNL
};

BatchNormHandle::~BatchNormHandle() {
#ifdef USE_DNNL
  if (use_dnnl) {
    delete (bn_fwd_training_d);
    delete (bn_fwd_training_pd);
  }
#endif  // USE_DNNL
}

#ifdef USE_DNNL

Tensor CpuBatchNormForwardInference(const BatchNormHandle& bnh, const Tensor& x,
                                    const Tensor& bnScale, const Tensor& bnBias,
                                    Tensor& running_mean, Tensor& running_var) {
  CHECK_EQ(x.device()->lang(), kCpp);
  Tensor y;
  y.ResetLike(x);

  Tensor w = get_bn_weight_from(bnScale, bnBias);

  y.device()->Exec(
      [y, w, x, &running_mean, &running_var, &bnh](Context* ctx) mutable {
        auto eng = ctx->dnnl_engine;
        using namespace dnnl;

        auto x_mem = memory(bnh.x_md, eng, x.block()->mutable_data());
        auto y_mem = memory(bnh.x_md, eng, y.block()->mutable_data());
        // indicates using scale&bias and running mean&var
        auto flags_ = normalization_flags::use_scale_shift |
                      normalization_flags::use_global_stats;

        auto bn_fwd_d = batch_normalization_forward::desc(
            prop_kind::forward_inference, bnh.x_md, bnh.epsilon, flags_);
        auto bn_fwd_pd =
            batch_normalization_forward::primitive_desc(bn_fwd_d, eng);
        auto m_mem = memory(bn_fwd_pd.mean_desc(), eng,
                            running_mean.block()->mutable_data());
        auto v_mem = memory(bn_fwd_pd.variance_desc(), eng,
                            running_var.block()->mutable_data());
        auto w_mem =
            memory(bn_fwd_pd.weights_desc(), eng, w.block()->mutable_data());

        // execution
        batch_normalization_forward(bn_fwd_pd).execute(
            ctx->dnnl_stream, {{DNNL_ARG_SRC, x_mem},
                               {DNNL_ARG_DST, y_mem},
                               {DNNL_ARG_SCALE_SHIFT, w_mem},
                               {DNNL_ARG_MEAN, m_mem},
                               {DNNL_ARG_VARIANCE, v_mem}});
        ctx->dnnl_stream.wait();
      },
      {x.block(), w.block(), running_mean.block(), running_var.block()},
      {y.block(), running_mean.block(), running_var.block()},
      "CpuBatchNormForwardInference");

  return y;
}

const std::vector<Tensor> CpuBatchNormForwardTraining(
    const BatchNormHandle& bnh, const Tensor& x, const Tensor& bnScale,
    const Tensor& bnBias, Tensor& running_mean, Tensor& running_var) {
  CHECK_EQ(x.device()->lang(), kCpp);
  Tensor y;
  y.ResetLike(x);

  // mean and var for local batch
  Tensor mean;
  mean.ResetLike(running_mean);
  Tensor var;
  var.ResetLike(running_var);

  // combine scale and bias to construct weight tensor in required format for
  // backward
  Tensor w = get_bn_weight_from(bnScale, bnBias);

  y.device()->Exec(
      [y, mean, var, w, x, &running_mean, &running_var,
       &bnh](Context* ctx) mutable {
        auto eng = ctx->dnnl_engine;
        using namespace dnnl;

        auto x_mem = memory(bnh.x_md, eng, x.block()->mutable_data());
        auto y_mem = memory(bnh.x_md, eng, y.block()->mutable_data());
        auto m_mem = memory(bnh.bn_fwd_training_pd->mean_desc(), eng,
                            mean.block()->mutable_data());
        auto v_mem = memory(bnh.bn_fwd_training_pd->variance_desc(), eng,
                            var.block()->mutable_data());
        auto w_mem = memory(bnh.bn_fwd_training_pd->weights_desc(), eng,
                            w.block()->mutable_data());

        batch_normalization_forward(*bnh.bn_fwd_training_pd)
            .execute(ctx->dnnl_stream, {{DNNL_ARG_SRC, x_mem},
                                        {DNNL_ARG_DST, y_mem},
                                        {DNNL_ARG_SCALE_SHIFT, w_mem},
                                        {DNNL_ARG_MEAN, m_mem},
                                        {DNNL_ARG_VARIANCE, v_mem}});
        ctx->dnnl_stream.wait();

        // local implemented running mean as mkldnn does not support it yet:
        // https://github.com/intel/mkl-dnn/issues/371
        // https://github.com/intel/mkl-dnn/issues/517
        // https://arxiv.org/pdf/1502.03167.pdf
        auto s = x.shape();
        s[1] = 1;
        float p = Product(s);  // for unbiased variance
        running_mean = running_mean * (1 - bnh.factor) + mean * bnh.factor;
        running_var =
            running_var * (1 - bnh.factor) + var * (p / (p - 1)) * bnh.factor;
      },
      {x.block(), w.block(), running_mean.block(), running_var.block()},
      {y.block(), running_mean.block(), running_var.block(), mean.block(),
       var.block()},
      "CpuBatchNormForwardTraining");

  return {y, mean, var};
}

const std::vector<Tensor> CpuBatchNormBackwardx(
    const BatchNormHandle& bnh, const Tensor& y, const Tensor& dy,
    const Tensor& x, const Tensor& bnScale, const Tensor& bnBias,
    const Tensor& mean, const Tensor& var) {
  CHECK_EQ(x.device()->lang(), kCpp);
  CHECK_EQ(y.device()->lang(), kCpp);
  CHECK_EQ(dy.device()->lang(), kCpp);
  CHECK_EQ(mean.device()->lang(), kCpp);
  CHECK_EQ(var.device()->lang(), kCpp);
  CHECK_EQ(bnScale.device()->lang(), kCpp);
  CHECK_EQ(bnBias.device()->lang(), kCpp);

  Tensor dx;
  dx.ResetLike(dy);

  // combine scale and bias to construct weight tensor in required format for
  // backward
  Tensor w = get_bn_weight_from(bnScale, bnBias);

  // Tensor dw(Shape{bnScale.Size(), 2});
  Tensor dw;
  dw.ResetLike(w);

  dx.device()->Exec(
      [w, dw, dx, dy, x, y, mean, var, &bnh](Context* ctx) mutable {
        auto eng = ctx->dnnl_engine;
        using namespace dnnl;

        auto x_mem = memory(bnh.x_md, eng, x.block()->mutable_data());
        auto dx_mem = memory(bnh.x_md, eng, dx.block()->mutable_data());
        auto y_mem = memory(bnh.x_md, eng, y.block()->mutable_data());
        auto dy_mem = memory(bnh.x_md, eng, dy.block()->mutable_data());

        auto m_mem = memory(bnh.bn_fwd_training_pd->mean_desc(), eng,
                            mean.block()->mutable_data());
        auto v_mem = memory(bnh.bn_fwd_training_pd->variance_desc(), eng,
                            var.block()->mutable_data());
        auto w_mem = memory(bnh.bn_fwd_training_pd->weights_desc(), eng,
                            w.block()->mutable_data());

        auto bn_bwd_d = batch_normalization_backward::desc(
            prop_kind::backward, bnh.x_md, bnh.x_md, bnh.epsilon,
            normalization_flags::use_scale_shift);
        auto bn_bwd_pd = batch_normalization_backward::primitive_desc(
            bn_bwd_d, eng, *bnh.bn_fwd_training_pd);

        auto dw_mem = memory(bn_bwd_pd.diff_weights_desc(), eng,
                             dw.block()->mutable_data());

        batch_normalization_backward(bn_bwd_pd).execute(
            ctx->dnnl_stream, {{DNNL_ARG_SRC, x_mem},
                               {DNNL_ARG_DIFF_SRC, dx_mem},
                               {DNNL_ARG_DIFF_DST, dy_mem},
                               {DNNL_ARG_MEAN, m_mem},
                               {DNNL_ARG_VARIANCE, v_mem},
                               {DNNL_ARG_DIFF_SCALE_SHIFT, dw_mem},
                               {DNNL_ARG_SCALE_SHIFT, w_mem}});
        ctx->dnnl_stream.wait();
      },
      {x.block(), dy.block(), mean.block(), var.block(), w.block(), y.block()},
      {dx.block(), dw.block()}, "CpuBatchNormBackwardx");

  singa::Tensor dbnScale(bnScale.shape());
  CopyDataToFrom(&dbnScale, dw, bnScale.Size(), 0, 0);
  singa::Tensor dbnBias(bnBias.shape());
  CopyDataToFrom(&dbnBias, dw, bnBias.Size(), 0, bnScale.Size());

  CHECK(dbnScale.nDim() == bnScale.nDim()) << "dbnScale ndim not match bnScale";
  CHECK(dbnBias.nDim() == bnBias.nDim()) << "dbnScale ndim not match bnScale";
  CHECK(dbnScale.shape()[0] == bnScale.shape()[0])
      << "dbnScale shape not match bnScale";
  CHECK(dbnBias.shape()[0] == bnBias.shape()[0])
      << "dbnBias shape not match bnBias";

  return {dx, dbnScale, dbnBias};
}

#endif  // USE_DNNL

#ifdef USE_CUDNN
CudnnBatchNormHandle::CudnnBatchNormHandle(const float momentum,
                                           const Tensor& input)
    : BatchNormHandle(momentum, input) {
  if (is_2d) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL;
    if (const char* env_p = std::getenv("CUDNN_BATCHNORM_ALG")) {
      std::string alg = std::string(env_p);
      std::transform(alg.begin(), alg.end(), alg.begin(), toupper);
      if (alg == "CUDNN_BATCHNORM_SPATIAL_PERSISTENT")
        mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
      LOG(INFO) << " CUDNN_BATCHNORM_ALG: " << alg;
    }
  }
  DataType dtype = input.data_type();
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&shape_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&param_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(shape_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), batchsize,
                                         channels, height, width));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(param_desc, CUDNN_TENSOR_NCHW,
                                         GetCudnnDataType(dtype), 1, channels,
                                         1, 1));
};

const std::vector<Tensor> GpuBatchNormForwardTraining(
    const CudnnBatchNormHandle& cbnh, const Tensor& x, const Tensor& bnScale,
    const Tensor& bnBias, Tensor& running_mean, Tensor& running_var) {
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(bnScale.device()->lang(), kCuda);
  CHECK_EQ(bnBias.device()->lang(), kCuda);
  CHECK_EQ(running_mean.device()->lang(), kCuda);
  CHECK_EQ(running_var.device()->lang(), kCuda);

  Tensor mean;
  Tensor var;
  mean.ResetLike(running_mean);
  var.ResetLike(running_var);

  Shape shape = x.shape();

  Tensor input(x);  // for unification of 2d and 4d cases.
  if (cbnh.is_2d) input.Reshape(Shape{shape.at(0), shape.at(1), 1, 1});

  Tensor output;
  output.ResetLike(x);

  output.device()->Exec(
      [=, &bnScale, &bnBias, &running_mean, &running_var,
       &cbnh](Context* ctx) mutable {
        const float alpha = 1.0f, beta = 0.0f;
        double epsilon = CUDNN_BN_MIN_EPSILON;
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            ctx->cudnn_handle, cbnh.mode, &alpha, &beta, cbnh.shape_desc,
            input.block()->data(), cbnh.shape_desc,
            output.block()->mutable_data(), cbnh.param_desc,
            bnScale.block()->data(), bnBias.block()->data(), cbnh.factor,
            running_mean.block()->mutable_data(),
            running_var.block()->mutable_data(), epsilon,
            mean.block()->mutable_data(), var.block()->mutable_data()));
      },
      {input.block(), bnScale.block(), bnBias.block(), running_mean.block(),
       running_var.block()},
      {output.block(), running_mean.block(), running_var.block(), mean.block(),
       var.block()},
      "GpuBatchNormForwardTraining");
  if (cbnh.is_2d) output.Reshape(Shape{shape.at(0), shape.at(1)});
  return {output, mean, var};
}

Tensor GpuBatchNormForwardInference(const CudnnBatchNormHandle& cbnh,
                                    const Tensor& x, const Tensor& bnScale,
                                    const Tensor& bnBias,
                                    const Tensor& running_mean,
                                    const Tensor& running_var) {
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(bnScale.device()->lang(), kCuda);
  CHECK_EQ(bnBias.device()->lang(), kCuda);
  CHECK_EQ(running_mean.device()->lang(), kCuda);
  CHECK_EQ(running_var.device()->lang(), kCuda);

  Shape shape = x.shape();

  Tensor input(x);  // for unification of 2d and 4d cases.
  if (cbnh.is_2d) input.Reshape(Shape{shape.at(0), shape.at(1), 1, 1});

  Tensor output;
  output.ResetLike(x);
  output.device()->Exec(
      [=, &bnScale, &bnBias, &running_mean, &running_var,
       &cbnh](Context* ctx) mutable {
        const float alpha = 1.0f, beta = 0.0f;
        double epsilon = CUDNN_BN_MIN_EPSILON;
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            ctx->cudnn_handle, cbnh.mode, &alpha, &beta, cbnh.shape_desc,
            input.block()->data(), cbnh.shape_desc,
            output.block()->mutable_data(), cbnh.param_desc,
            bnScale.block()->data(), bnBias.block()->data(),
            running_mean.block()->data(), running_var.block()->data(),
            epsilon));
      },
      {input.block(), bnScale.block(), bnBias.block(), running_mean.block(),
       running_var.block()},
      {output.block()}, "GpuBatchNormForwardInference");
  return output;
}

const std::vector<Tensor> GpuBatchNormBackward(
    const CudnnBatchNormHandle& cbnh, const Tensor& dy, const Tensor& x,
    const Tensor& bnScale, const Tensor& mean, const Tensor& var) {
  CHECK_EQ(dy.device()->lang(), kCuda);
  CHECK_EQ(x.device()->lang(), kCuda);
  CHECK_EQ(bnScale.device()->lang(), kCuda);
  CHECK_EQ(mean.device()->lang(), kCuda);
  CHECK_EQ(var.device()->lang(), kCuda);

  Tensor dx;
  dx.ResetLike(dy);

  Tensor dbnScale;
  dbnScale.ResetLike(bnScale);

  Tensor dbnBias;
  dbnBias.ResetLike(bnScale);

  dx.device()->Exec(
      [=, &bnScale, &cbnh](Context* ctx) mutable {
        const float alpha = 1.0f, beta = .0f;
        double epsilon = CUDNN_BN_MIN_EPSILON;
        CUDNN_CHECK(cudnnBatchNormalizationBackward(
            ctx->cudnn_handle, cbnh.mode, &alpha, &beta, &alpha, &beta,
            cbnh.shape_desc, x.block()->data(), cbnh.shape_desc,
            dy.block()->data(), cbnh.shape_desc, dx.block()->mutable_data(),
            cbnh.param_desc, bnScale.block()->data(),
            dbnScale.block()->mutable_data(), dbnBias.block()->mutable_data(),
            epsilon, mean.block()->data(), var.block()->data()));
      },
      {x.block(), dy.block(), bnScale.block(), mean.block(), var.block()},
      {dx.block(), dbnScale.block(), dbnBias.block()}, "GpuBatchNormBackward");

  if (cbnh.is_2d) dx.Reshape(Shape{dx.shape().at(0), dx.shape().at(1)});

  return {dx, dbnScale, dbnBias};
}

#endif  // USE_CUDNN
}  // namespace singa

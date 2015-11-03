#ifndef MATH_KERNEL_H
#define MATH_KERNEL_H

namespace singa{

extern "C" {
	void singa_gpu_sum_vec(float *data, float *sum , long n);

	void singa_gpu_sum_col(const float *src_mat_data, float *dst_vec_data, long rows, long cols, long stride);

	void singa_gpu_add_vec_row(const float *src_vec_data, const float *src_mat_data, float *des_mat_data, long rows, long cols, long stride);

	void singa_gpu_set_value(float *data, float value, long n);

	void singa_gpu_scale(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_scale_grad(float *data, float alpha, long n);

	void singa_gpu_exp(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_exp_grad(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_sigmoid(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_sigmoid_grad(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_relu(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_relu_grad(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_tanh(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_tanh_grad(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_softplus(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_softplus_grad(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_square(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_square_grad(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_sqrt(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_threshold(const float *src_data, float *des_data, float alpha, long n);

	void singa_gpu_add(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n);

	void singa_gpu_sub(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n);

	void singa_gpu_mult(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n);

	void singa_gpu_div(const float *src_data_a, const float *src_data_b, float *des_data, float alpha, float beta, long n);

};

}

#endif

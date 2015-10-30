#ifndef MATH_KERNEL_H
#define MATH_KERNEL_H

namespace singa{

extern "C" {
	void singa_sum_col(float *src_mat_data, float *dst_vec_data, long rows, long cols, long stride);

	void singa_add_vec_row(float *src_vec_data, float *src_mat_data, float *des_mat_data, long rows, long cols, long stride);
};

}

#endif

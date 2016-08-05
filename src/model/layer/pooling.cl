__kernel void max_pool_forward(
    const int nthreads, __global const float* bottom_data, const int channels, 
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    __global float* top_data, __global int* mask) {

  for (int i = get_global_id(0); i < nthreads; i += get_global_size(0)) {
    const int pw = i % pooled_width;
    const int ph = (i / pooled_width) % pooled_height;
    const int c = (i / pooled_width / pooled_height) % channels;
    const int n = i / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, (int)0);
    wstart = max(wstart, (int)0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    __global const float* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[i] = maxval;
    mask[i] = maxidx;
  }
}

__kernel void ave_pool_forward(
    const int nthreads, __global const float* const bottom_data, const int channels, 
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, 
    const int pad_h, const int pad_w, __global float* top_data) {
    
  for (int i = get_global_id(0); i < nthreads; i += get_global_size(0)) {
    {
      const int pw = i % pooled_width;
      const int ph = (i / pooled_width) % pooled_height;
      const int c = (i / pooled_width / pooled_height) % channels;
      const int n = i / pooled_width / pooled_height / channels;
      int hstart = ph * stride_h - pad_h;
      int wstart = pw * stride_w - pad_w;
      int hend = min(hstart + kernel_h, height + pad_h);
      int wend = min(wstart + kernel_w, width + pad_w);
      const int pool_size = (hend - hstart) * (wend - wstart);
      hstart = max(hstart, (int)0);
      wstart = max(wstart, (int)0);
      hend = min(hend, height);
      wend = min(wend, width);
      float aveval = 0;
      __global const float* bottom_slice = bottom_data
          + (n * channels + c) * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          aveval += bottom_slice[h * width + w];
        }
      }
      top_data[i] = aveval / pool_size;
    }
  }
}

__kernel void sto_pool_forward_train(
    const int nthreads, __global const float* bottom_data,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
    __global float* rand_idx,
    __global float* top_data) {
    
  for (int i = get_global_id(0); i < nthreads;
      i += get_global_size(0)) {
    const int pw = i % pooled_width;
    const int ph = (i / pooled_width) % pooled_height;
    const int c = (i / pooled_width / pooled_height) % channels;
    const int n = i / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    float cumsum = 0.;
    __global const float* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[i] * cumsum;
    // Second pass: get value, and set i.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[i] = ((n * channels + c) * height + h) * width + w;
          top_data[i] = bottom_slice[h * width + w];
          h = hend;
          w = wend;
        }
      }
    }
  }
}

__kernel void sto_pool_forward_test(
    const int nthreads, __global const float* const bottom_data, const int channels, 
    const int height, const int width,
    const int pooled_height, const int pooled_width, 
    const int kernel_h, const int kernel_w, 
    const int stride_h, const int stride_w,
    __global float* top_data) {
    
  for (int i = get_global_id(0); i < nthreads;
      i += get_global_size(0)) {
    const int pw = i % pooled_width;
    const int ph = (i / pooled_width) % pooled_height;
    const int c = (i / pooled_width / pooled_height) % channels;
    const int n = i / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    float cumsum = FLT_MIN;
    float cumvalues = 0.;
    __global const float* bottom_slice = bottom_data
        + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[i] = cumvalues / cumsum;
  }
}

__kernel void max_pool_backward(const int nthreads,
                                __global const float* top_diff,
                                const int use_mask,
                                __global const int* mask,
                                __global const float* top_mask,
                                const int channels,
                                const int height,
                                const int width,
                                const int pooled_height,
                                const int pooled_width,
                                const int kernel_h,
                                const int kernel_w,
                                const int stride_h,
                                const int stride_w,
                                const int pad_h,
                                const int pad_w,
                                __global float* bottom_diff) {
  for (int i = get_global_id(0); i < nthreads;
      i += get_global_size(0)) {
    // find out the local i
    // find out the local offset
    const int w = i % width;
    const int h = (i / width) % height;
    const int c = (i / width / height) % channels;
    const int n = i / width / height / channels;
    const int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    float gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    __global const float* top_diff_slice = top_diff + offset;
    if (use_mask == 1) {
      __global const int* mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      __global const float* top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[i] = gradient;
  }
}

__kernel void ave_pool_backward(const int nthreads,
                                __global const float* top_diff,
                                const int channels,
                                const int height,
                                const int width,
                                const int pooled_height,
                                const int pooled_width,
                                const int kernel_h,
                                const int kernel_w,
                                const int stride_h,
                                const int stride_w,
                                const int pad_h,
                                const int pad_w,
                                __global float* bottom_diff) {
  for (int i = get_global_id(0); i < nthreads;
      i += get_global_size(0)) {
    // find out the local i
    // find out the local offset
    const int w = i % width + pad_w;
    const int h = (i / width) % height + pad_h;
    const int c = (i / width / height) % channels;
    const int n = i / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    float gradient = 0.0;
    __global const float* const top_diff_slice = top_diff
        + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[i] = gradient;
  }
}

__kernel void sto_pool_backward(
    const int nthreads, __global const float* rand_idx,
    __global const float* const top_diff, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    __global float* bottom_diff) {

  for (int i = get_global_id(0); i < nthreads; i +=
      get_global_size(0)) {
    // find out the local i
    // find out the local offset
    const int w = i % width;
    const int h = (i / width) % height;
    const int c = (i / width / height) % channels;
    const int n = i / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    float gradient = 0.0;
    __global const float* rand_idx_slice = rand_idx
        + (n * channels + c) * pooled_height * pooled_width;
    __global const float* top_diff_slice = top_diff
        + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw]
            * (i == (int) (rand_idx_slice[ph * pooled_width + pw])?1.0:0.0);
      }
    }
    bottom_diff[i] = gradient;
  }
}


#ifndef SINGA_UTILS_MKLDNN_UTILS_H_
#define SINGA_UTILS_MKLDNN_UTILS_H_

#include <mkldnn.hpp>

namespace singa {
  /*
   supported data type by mkldnn
   mkldnn_f32 - 32-bit/single-precision floating point.
   mkldnn_s32 - 32-bit signed integer.
   mkldnn_s16 - 16-bit signed integer.
   mkldnn_s8 - 8-bit signed integer.
   mkldnn_u8 - 8-bit unsigned integer.
   */
  inline mkldnn::memory::data_type GetMKLDNNDataType(DataType dtype) {
    mkldnn::memory::data_type ret = mkldnn::memory::data_type::f32;
    switch (dtype) {
      case kFloat32:
        ret = mkldnn::memory::data_type::f32;
        break;
      case kDouble:
        LOG(FATAL) << "The data type " << DataType_Name(dtype)
                   << " is not support by mkldnn";
        break;
      case kFloat16:
        LOG(FATAL) << "The data type " << DataType_Name(dtype)
                   << " is not support by mkldnn";
        break;
      default:
        LOG(FATAL) << "The data type " << DataType_Name(dtype)
                   << " is not support by mkldnn";
    }
    return ret;
  }
}
#endif // SINGA_UTILS_MKLDNN_UTILS_H_

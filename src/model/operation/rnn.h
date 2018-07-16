class RecHandle {
public:
  RecHandle(const size_t Input_size, const size_t Hidden_size, const size_t Num_stacks,
            const std::string nonlinearity, const bool bias, const float dropout, const bool bidirectional);

  size_t input_size_;
  size_t hidden_size_;
  size_t num_stacks_;
  float dropout_;
  size_t seed_ = 0x1234567;
  size_t num_directions_;
  std::string rnn_mode_;
  bool has_cell;
  size_t weight_size;

  size_t batch_size = 0;
  size_t seq_length_ = 0;
  size_t max_length_ = 0;
}

class CudnnRecHandle: public RecHandle {
public:
  CudnnRecHandle(const vector<Tensor> &inputs，const size_t Input_size, const size_t Hidden_size, const size_t Num_stacks,
                 const std::string nonlinearity, const bool bias, const float dropout, const bool bidirectional);
  void UpdateStates(size_t num_x, const vector<Tensor> &inputs);
  void UpdateIODescriptors(size_t len, const vector<Tensor> &inputs);
  void ResetHiddenAndCellDescriptors(size_t batch_size);
  void SetRNNDescriptor(shared_ptr<Device> dev);
  void UpdateSpaces(size_t seq_length, shared_ptr<Device> dev);

  cudnnTensorDescriptor_t* x_descs_ = nullptr;
  cudnnTensorDescriptor_t* dx_descs_ = nullptr;
  cudnnTensorDescriptor_t* y_descs_ = nullptr;
  cudnnTensorDescriptor_t* dy_descs_ = nullptr;
  cudnnTensorDescriptor_t hx_desc_ = nullptr;
  cudnnTensorDescriptor_t dhx_desc_ = nullptr;
  cudnnTensorDescriptor_t cx_desc_ = nullptr;
  cudnnTensorDescriptor_t dcx_desc_ = nullptr;
  cudnnTensorDescriptor_t hy_desc_ = nullptr;
  cudnnTensorDescriptor_t dhy_desc_ = nullptr;
  cudnnTensorDescriptor_t cy_desc_ = nullptr;
  cudnnTensorDescriptor_t dcy_desc_ = nullptr;
  cudnnFilterDescriptor_t weight_desc_ = nullptr;
  cudnnFilterDescriptor_t dweight_desc_ = nullptr;
  cudnnRNNDescriptor_t rnn_desc_ = nullptr;
  cudnnDropoutDescriptor_t dropout_desc_ = nullptr;
  cudnnDataType_t dtype_ = CUDNN_DATA_FLOAT;
  Tensor workspace_;
  Tensor reserve_space_;
  Tensor dropout_state_;
}

Tensor MergeInputs(size_t num, const vector<Tensor> &in);

vector<Tensor> SplitOutput(size_t num, size_t dim,
                                     const vector<Tensor> &in,
                                     const Tensor output);

std::vector<std::vector<Tensor>> GpuRecForwardTraining(const CudnnRecHandle &crh, const vector<Tensor> &inputs， const Tensor &W);

std::vector<Tensor> GpuRecForwardInference(const CudnnRecHandle &crh, const vector<Tensor> &inputs，const Tensor &W);

const std::pair<vector<Tensor>, Tensor> GpuRecBackward(const CudnnRecHandle &crh, const vector<Tensor> &grads， const vector<Tensor> &cache);

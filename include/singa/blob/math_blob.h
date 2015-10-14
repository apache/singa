/*********************Level-2 interface, called by user code*******************/
// c++ ususally use const & for input arguments, and * for output arguments.
// ww: maybe we represent Blob's shape using int s[4]+dim? currently we use a vector, which may
// not be convenient as int array.

/**********************************************************************************/
// class1 matrix operation

void MMDot(const Blob & A, const Blob & B, Blob & C);
// A,B and C are matrix

void MVDot(const Blob & A, const Blob & B, Blob & C);
// A is matrix,B and C are vector

void VVDot(const Blob & A, const Blob & B, Blob & C);
// C is matrix,A and B are vector

float VVdot(const Blob & A, const Blob & B);
//A and B are vectors

void GEMM(const Blob & A, const Blob & B, Blob & C, float alpha = 1, float beta = 1);
//C = alpha*A*B+beta*C, A, B and C are matrix

Blob Reshape(const Blob & A, const std::vector<int>& shape);
// the current reshape in blob.h is: void Reshape(const std::vector<int>& shape);
// return the reference of the reshaped blob

Blob Transpose(const Blob & A);
// return A^T, only reference to the blob A
// ww: just add a bool field in Blob, e.g., transpose_
/**********************************************************************************/
// class2 element-wise operation

void Set(Blob & A,float alpha);
// element-wise operation: Ai = alpha

void AXPY(const Blob & A, Blob & B, float alpha);
// element-wise operation: Bi = alpha*Ai+Bi  A and B should have the same size

void Add(const Blob & A, const Blob & B, Blob & C);
// element-wise operation: Ci = Ai+Bi  A,B and C should have the same size

void Sub(const Blob & A, const Blob & B, Blob & C);
// element-wise operation: Ci = Ai-Bi  A,B and C should have the same size

void Mult(const Blob & A, const Blob & B, Blob & C);
// element-wise operation: Ci = Ai*Bi  A,B and C should have the same size

void Div(const Blob & A, const Blob & B, Blob & C);
// element-wise operation: Ci = Ai/Bi  A,B and C should have the same size

void Scale(const Blob & A, Blob & B, float alpha);
// element-wise operation: Bi = alpha*Ai

void Sigmoid(const Blob & A, Blob & B,float t);
// element-wise operation: Bi = 1/(1+e^(-Ai*t))

void Relu(const Blob & A, Blob & B,float t = 0);
// element-wise operation: Bi = ((1-t)abs(Ai) + (1+t)Ai)/2

void Tanh(const Blob & A, Blob & B,float t);
// element-wise operation: Bi = tanh(Ai*t)

void Exp(const Blob & A, Blob & B, float alpha = 2.71);
// element-wise operation: Bi = alpha^Ai
// ww: there are many element-wise operations, e.g., log, square, sqrt. 
// If MKL/OpenBLAS or other libraries do not optimize these operations in ad-hoc manner, 
// then we can implement then using the E_Func below by passing the basic log/square/sqrt operations.



template<typename Op> void E_Func(Blob & A, float alpha);
template<typename Op> void E_Func(const Blob & A, Blob & B, float alpha, float beta);
template<typename Op> void E_Func(const Blob & A, const Blob & B, Blob & C, float alpha, float beta);
// element-wise generalized operation defined in Op


// ww: the following functions may require thread specific variables, e.g., seed or random stream state.

void Gaussian(Blob & A, float mu, float sigma);
// element-wise operation: initialize each element in A following distribution Gaussian(mu, sigma)

void Uniform(Blob & A, float low, float high);
// element-wise operation: initialize each element in A following uniform distribution from low to high

void Bernoulli(Blob & A, float p, int n = 1);
// element-wise operation: initialize each element in A following distribution Bernoulli(n,p)


/**********************************************************************************/
//class3 matrix-vector expand/reduce operation

template<typename Op> void Reduce_F(const Blob & A, Blob & B);
//reduce each row of A to an element of B e.g. the sum operation in softmax
template<typename Op> void Expand_F(const Blob & A, Blob & B);
//expand each element in A into a row of B

void Repmat(const Blob & A, Blob & B);
// A is a vector, B is a matrix , let each row of B to be A
// just copy memory, will be faster

// ww may rename to MVAdd, MVSum to be consistent with the MVDot, MMDot, VVDot.
void MVAdd(const Blob & A, Blob & B, float alpha, float beta);
// A is a vector, B is a matrix , Bij = alpha*Ai+beta*Bij
// will use gemm. faster than general expand_f

void MVSum(const Blob & A, Blob & B, float alpha, float beta);
// A is a vector, B is a matrix , Ai = \sigma_j_{alpha*Bij}+beta*Ai
// will use gemm. faster than general reduce_f

void Softmax(const Blob & A,Blob & B,float alpha);
// Bij = e^(alpha*Aij) / \sigma_i_{e^(alpha*Aij)}

/**********************************************************************************/
//class4 convolution operation

void Conv(const Blob & A,const Blob & B,Blob & C);
// A is the data, B is the parameter, C is the result

void Pool(const Blob & A,Blob & B, int method);
// A is the data, B is the result, should indicate max or ave pooling

// jy: need to provide grad compute function respectively?

// ww: The conv/pool operations cannot be declared as above, 
// because they require other parameters, e.g., filter size, num of filters, pad, stride, etc.
// Caffe and mxnet use cudnn in layer implementations instead of implementing low-level operations.
// Maybe we can follow caffe? layer implementation is not our contribution, we can use others' code directly.
// For CPU version, we use im2col and col2im for the conv layer. 
// For GPU version, we use cudnn for the conv layer. Similarly for other layers. 
// In conclude, we may not implement Blob level Conv and Pool operations. 
// Instead, we implement CaffeConvLayer, CaffePoolLayer, cudnnConvLayer, cudnnPoolLayer.
// Later we may add IntelConvLayer (cpu), NeonConvLayer (gpu).

Blob setcolspace(const Blob & A);
void im2col(const Blob & A,Blob & B);
void col2im(const Blob & A,Blob & B);
//given an img, use setcolspace to generate colspace Blob
//use pack/unpack to get data in col/img

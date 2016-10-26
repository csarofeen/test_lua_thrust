#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/random.h>

#include <iostream>

#include <cuda_fp16.h>

template <typename T>
struct asum_amax_type
{
  T asum_val;
  T amax_val;
  int nnz;
};

template <typename T>
struct asum_amax_binary_op
  : public thrust::binary_function< asum_amax_type<T>, asum_amax_type<T>, asum_amax_type<T> >
{
    __host__ __device__
    asum_amax_type<T> operator()(const asum_amax_type<T>& x, const asum_amax_type<T>& y) const
  {
    asum_amax_type<T> result;
    result.nnz = x.nnz + y.nnz;
    result.asum_val = x.asum_val + y.asum_val;
    result.amax_val = thrust::max(x.amax_val, y.amax_val);
    return result;
  }
};

struct h2f_unary_op
  : public thrust::unary_function<unsigned short, float>
{
  __device__
  asum_amax_type<float> operator()(const unsigned short& x) const
  {
    half val = *( (half*) &x);
    asum_amax_type<float> result;
    result.asum_val = fabsf(__half2float(val));
    result.amax_val = result.asum_val;
    result.nnz = (result.asum_val == 0.f) ? 0 : 1;
    return result;
  }
};

typedef struct float_pair{
  float aave;
  float amax;
} float_pair_t;

extern "C"
float_pair_t fp16_stats(half* d_data, int N){
  if((uintptr_t)(const void *)(d_data) % 4 == 0) std::cout<<"Aligned at 4Byte boundary"<<std::endl;
  else if( (uintptr_t)(const void *)(d_data) % 2 == 0) std::cout<<"Aligned at 2Byte boundary"<<std::endl;
  if(N%2 != 0){
    std::cerr<<"Odd sized tensors are not supported at the moment"<<std::endl;
    throw(-1);
  }

  thrust::device_ptr<unsigned short> d_ptr = thrust::device_pointer_cast((unsigned short*)d_data);

  h2f_unary_op unary_op;
  asum_amax_binary_op<float> binary_op;

  asum_amax_type<float> init;
  init.amax_val = 0;
  init.nnz=0;
  init.asum_val = 0;

  asum_amax_type<float> result = thrust::transform_reduce(d_ptr, d_ptr+N, unary_op, init, binary_op);
  float_pair_t return_result;
  return_result.aave = result.asum_val/(float)result.nnz;
  return_result.amax = result.amax_val;
  std::cout<<return_result.aave<<std::endl;
  std::cout<<return_result.amax<<std::endl;
  return return_result;

}



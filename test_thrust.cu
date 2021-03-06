#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/random.h>

#include <iostream>

#include <cuda.h>
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
  : public thrust::unary_function<unsigned int, float>
{
  __device__
  asum_amax_type<float> operator()(const unsigned int& x) const
  {
    half2 val = *( (half2*) &x);
    float2 fval = __half22float2(val);
    fval.x = fabsf(fval.x);
    fval.y = fabsf(fval.y);
    
    asum_amax_type<float> result;

    result.nnz = (fval.x == 0.f) ? 0 : 1;
    result.nnz = (fval.y == 0.f) ? result.nnz : result.nnz + 1;
    result.asum_val = fval.x+fval.y;
    
    result.amax_val = thrust::max(fval.x, fval.y);

    return result;
  }
};

typedef struct float_pair{
  float aave;
  float amax;
} float_pair_t;

extern "C"
float_pair_t half2_stats(half* d_data, int N){
  if((uintptr_t)(const void *)(d_data) % 4 == 0) std::cout<<"Aligned at 4Byte boundary"<<std::endl;
  else if( (uintptr_t)(const void *)(d_data) % 2 == 0) std::cout<<"Aligned at 2Byte boundary"<<std::endl;
  if(N%2 != 0){
    std::cerr<<"Odd sized tensors are not supported at the moment"<<std::endl;
    throw(-1);
  }

  float time;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  thrust::device_ptr<unsigned int> d_ptr = thrust::device_pointer_cast((unsigned int*)d_data);

  h2f_unary_op unary_op;
  asum_amax_binary_op<float> binary_op;

  asum_amax_type<float> init;
  init.amax_val = 0;
  init.nnz=0;
  init.asum_val = 0;

  asum_amax_type<float> result = thrust::transform_reduce(d_ptr, d_ptr+(N/2), unary_op, init, binary_op);
  float_pair_t return_result;
  return_result.aave = result.asum_val/(float)result.nnz;
  return_result.amax = result.amax_val;


  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  printf("Time to reduce %f GB:  %f ms \n", ((float)N*2)/(1024*1024*1024), time);
  printf("Bandwidth is: %f GB/s \n", ((float)N*2)/(time/1000)/(1024*1024*1024) );
  
  return return_result;

}

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <algorithm>
#include <cstdlib>

extern "C"
float test(float* array, int N){
  thrust::device_vector<float> d_ptr(array, array+N);
  return thrust::reduce(d_ptr.begin(), d_ptr.end(), 0, thrust::plus<float>() );
}

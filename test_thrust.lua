
ffi = require("ffi")
require ('cutorch')


myLib = ffi.load("libtest_thrust.so")

a = torch.rand(20000,50000):cuda()
a = a - 0.6
a = a:type("torch.CudaHalfTensor")

a[1][1] = 0
a[1][2] = 0
a[2][3] = 0

--print(a)

nVals = a:storage():size()

assert(a:isContiguous(), "Tensor must be contiguous to use with thrust")


ffi.cdef("typedef struct float_pair{float aave; float amax;} float_pair_t;")
ffi.cdef("float_pair_t half2_stats(half *d_data, int N)")
float_pair_type = ffi.typeof("float_pair_t")

float_pair = float_pair_type()
float_pair = myLib.half2_stats(a:data(), nVals)

print("average of abs non-zero values: "..float_pair.aave)
print("maximum abs value: "..float_pair.amax)

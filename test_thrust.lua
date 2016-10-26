ffi = require("ffi")
require ('cutorch')


myLib = ffi.load("libtest_thrust.so")

buf = ffi.new("float[?]", 10)
for i=0,9 do buf[i]=i+2 end
io.write("Summing with thrust: ")
for i=0,9 do io.write(buf[i].." ") end
io.write("\n")
ffi.cdef("float test(float* array, int N);")
ffi.cdef("float sum_dev_float(float* d_data, int N);")

print( "result is: "..myLib.test(buf,10) )


a = torch.rand(2,5,3):cuda()

a[1][1][2] = 0
a[2][4][1] = 0
a[1][2][3] = 0

print(a)

nVals = a:storage():size()

assert(a:isContiguous(), "Tensor must be contiguous to use with thrust")

print("Sum of a is: "..myLib.sum_dev_float(a:data(), nVals) )

ffi.cdef("typedef struct float_pair{float aave; float amax;} float_pair_t;")
ffi.cdef("float_pair_t get_stats(float *d_data, int N);")

float_pair_type = ffi.typeof("float_pair_t")
float_pair = float_pair_type()
float_pair = myLib.get_stats(a:data(), nVals)
print("average of abs: "..float_pair.aave)
print("max abs value: "..float_pair.amax)

a = a:type("torch.CudaHalfTensor")

ffi.cdef("float_pair_t fp16_test(half *d_data, int N)")

float_pair = myLib.fp16_test(a:data(), nVals)
print("average of abs: "..float_pair.aave)
print("max abs value: "..float_pair.amax)

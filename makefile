testlib: test_thrust.cu
	nvcc --compiler-options '-fPIC' -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60 --shared -o  "libtest_thrust.so" test_thrust.cu

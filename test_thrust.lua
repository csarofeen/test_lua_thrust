ffi = require("ffi")
myLib = ffi.load("libtest_thrust.so")

buf = ffi.new("float[?]", 10)
for i=0,9 do buf[i]=i+2 end
io.write("Summing with thrust: ")
for i=0,9 do io.write(buf[i].." ") end
io.write("\n")
ffi.cdef("float test(float* array, int N);")
print( "result is: "..myLib.test(buf,10) )

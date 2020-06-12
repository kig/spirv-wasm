cd ispc_ivec4 &&
cpp httpd_ivec4.glsl | node ../preprocess.js > out_int.comp &&
glslangValidator -V -o httpd.spv out_int.comp &&
spirv-cross-ispc --ispc --output httpd.ispc httpd.spv &&
ispc -O3 --target=avx2-i64x4 -o httpd.ispc.o runner_ivec4.ispc &&
clang++ -pthread -std=c++11 -lm -pthread -O3 -o httpd_ivec4 httpd.ispc.o ../tasksys.cpp httpd_ispc_ivec4.cpp

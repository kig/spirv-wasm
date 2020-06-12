cd cpp &&
cpp httpd_ivec4.glsl | node ../preprocess.js > out.comp &&
glslangValidator -V -o httpd.spv out.comp &&
spirv-cross --cpp --output httpd.cpp httpd.spv &&
clang++ -lpthread -lm -O2 -o cppRunner httpd.cpp cppRunner.cpp -std=c++11

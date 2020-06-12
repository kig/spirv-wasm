cd cpp &&
cpp httpd_ivec4.glsl | node ../preprocess.js > out.comp &&
glslangValidator -V -o httpd.spv out.comp &&
spirv-cross --cpp --output httpd.cpp httpd.spv &&
clang++ -lpthread -I../../include -lm -O3 -o cppRunner cppRunner.cpp -std=c++11

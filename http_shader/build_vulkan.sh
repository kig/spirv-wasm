cpp httpd_ivec4.glsl | node preprocess.js > out.comp &&
glslangValidator -V -o httpd.spv out.comp &&
clang++ -lvulkan -lpthread -m64 -O2 -o vulkanRunner vulkanRunner.cpp -std=c++11

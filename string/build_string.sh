cpp test_string.glsl | node preprocess.js > out_string.comp &&
glslangValidator -V -o vulkan/test_string.spv out_string.comp &&
cd vulkan &&
clang++ -llz4 -lzstd -lvulkan -lpthread -m64 -O2 -o test_string test_string.cpp -std=c++17

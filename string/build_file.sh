cpp test_file.glsl | node preprocess.js > out_file.comp &&
glslangValidator -V -o vulkan/test_file.spv out_file.comp &&
cd vulkan &&
clang++ -lzstd -llz4 -lvulkan -lpthread -m64 -O2 -o test_file test_file.cpp -std=c++17

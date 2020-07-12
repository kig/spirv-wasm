glsl2spv test_string.glsl vulkan/test_string.spv &&
cd vulkan &&
clang++ -llz4 -lzstd -lvulkan -lpthread -m64 -O2 -o test_string test_string.cpp -std=c++17

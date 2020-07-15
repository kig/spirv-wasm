cd vulkan &&
clang++ -ldl -llz4 -lzstd -lvulkan -lpthread -m64 -O2 -march=znver1 -mtune=znver1 -o gls gls.cpp -std=c++17

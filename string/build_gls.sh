cd vulkan &&
clang++ -llz4 -lzstd -lvulkan -lpthread -m64 -O2 -march=skylake -mtune=skylake -o gls gls.cpp -std=c++17

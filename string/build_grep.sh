cpp grep.glsl | node preprocess.js > out_file.comp &&
glslangValidator -V -o vulkan/grep.spv out_file.comp &&
cd vulkan &&
clang++ -llz4 -lzstd -lvulkan -lpthread -m64 -O3 -march=znver1 -mtune=znver1 -o grep grep.cpp -std=c++17

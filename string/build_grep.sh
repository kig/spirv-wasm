cpp grep.glsl | node preprocess.js > out_file.comp &&
glslangValidator -V -o vulkan/grep.spv out_file.comp &&
cd vulkan &&
clang++ -lvulkan -lpthread -m64 -O2 -march=skylake -mtune=skylake -o grep grep.cpp -std=c++11

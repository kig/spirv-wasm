cpp test_string.glsl | node preprocess.js > out.comp &&
glslangValidator -V -o vulkan/test_string.spv out.comp &&
cd vulkan &&
clang++ -lvulkan -lpthread -m64 -O2 -o vulkanRunner vulkanRunner.cpp -std=c++11

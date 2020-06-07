# SPIRV-Cross-ISPC generates buggy varying/uniform typings, httpd.ispc needs to be manually fixed after compiles.
# So let's not overwrite the working version on every build.

#cpp httpd_int.glsl | node preprocess.js > out.comp &&
#glslangValidator -V -o httpd.spv out.comp &&
#spirv-cross-ispc --ispc --output httpd.ispc httpd.spv &&
ispc -O3 -o httpd.ispc.o runner.ispc &&
clang++ -I/usr/local/bin -pthread -std=c++11 -lm -pthread -O3 -o httpd httpd.ispc.o tasksys.cpp httpd_ispc.cpp

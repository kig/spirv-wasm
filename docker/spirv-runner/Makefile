TARGET := TEST

builtins.o: /usr/local/src/ispc-wasm/builtins/builtins.c
	emcc -DWASM_IMPLEMENTATION /usr/local/src/ispc-wasm/builtins/builtins.c -O3 -s EXPORTED_FUNCTIONS='["___wasm_do_print", "___wasm_clock"]' -c -o builtins.o

tasksys.o: /usr/local/src/ispc-wasm/examples/tasksys.cpp
	emcc -DWASM -DISPC_USE_PTHREADS -s USE_PTHREADS=1 /usr/local/src/ispc-wasm/examples/tasksys.cpp -I./ -O2 -c -o tasksys.o

ispcRunner.o: ispcRunner.cpp
	emcc -DWASM ispcRunner.cpp -I./ -O3 -s EXPORTED_FUNCTIONS='["_main", "_run"]' -c -o ispcRunner.o -msimd128 -s SIMD=1

runner.ispc.o: program.comp.glsl
	glslangValidator -V -o program.spv program.comp.glsl
	spirv-cross-ispc --ispc --output program.ispc.raw program.spv
	ruby fix_ispc_input_output_order < program.ispc.raw > program.ispc
	ispc runner.ispc -O2 --target=wasm-i32x4 -o runner.ispc.o

build: builtins.o tasksys.o ispcRunner.o runner.ispc.o
	emcc -O3 ispcRunner.o builtins.o tasksys.o runner.ispc.o -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=64 -s TOTAL_MEMORY=268435456 -o $(TARGET).html -msimd128 -s SIMD=1 -s EXPORTED_FUNCTIONS='["_main", "_run"]' -s EXTRA_EXPORTED_RUNTIME_METHODS='["ccall"]'

all: builtins.o tasksys.o ispcRunner.o runner.ispc.o build

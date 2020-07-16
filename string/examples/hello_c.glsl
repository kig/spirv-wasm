#include <file.glsl>
#include <dlopen.glsl>

ThreadLocalCount = 1;
ThreadGroupCount = 1;

writeSync("hello.c", "#include <stdio.h>\nvoid hello(char* s){printf(\"Hello, %s!\\n\",s);}\nvoid sub(int* v, unsigned int vlen, int* res, unsigned int reslen) { res[0] = v[0]-v[1]; }");
awaitIO(runCmd("cc --shared -o hello.so hello.c"));
uint64_t lib = dlopenSync("./hello.so");
dlcallSync(lib, "hello", "GLSL\u0000", string(-4,-4));
alloc_t params = malloc(8);
i32heap[params.x/4] = 7;
i32heap[params.x/4+1] = 12;
alloc_t res = dlcallSync(lib, "sub", params, malloc(4));
int32_t subResult = readI32heap(res.x);
println(concat(str(i32heap[params.x/4]), " - ", str(i32heap[params.x/4+1]), " = ", str(subResult)));

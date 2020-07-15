#include <file.glsl>
#include <dlopen.glsl>

ThreadLocalCount = 1;
ThreadGroupCount = 1;

writeSync("hello.c", "#include <stdio.h>\nvoid hello(char* s){printf(\"Hello, %s!\\n\",s);}");
awaitIO(runCmd("cc --shared -o hello.so hello.c"));
uint64_t lib = dlopenSync("./hello.so");
dlcallSync(lib, "hello", "GLSL\u0000", 0);

#!/usr/bin/env gls

#include <file.glsl>
#include <dlopen.glsl>

ThreadLocalCount = 1;
ThreadGroupCount = 1;

uint64_t lib = 0;

int32_t dlcallSync_i32(uint64_t lib, string func, int32_t p0, int32_t p1) {
    int32_t r = 0;
    FREE_ALL(
        alloc_t params = malloc(8, 4);
        i32heap[params.x/4] = p0;
        i32heap[params.x/4 + 1] = p1;
        alloc_t res = dlcallSync(lib, func, params, malloc(4));
        r = readI32heap(res.x);
    );
    return r;
}

#define DLFUNC_I32_I32_I32(lib, func) int32_t func (int32_t p0, int32_t p1) { return dlcallSync_i32(lib, #func, p0, p1); }
#define DLFUNC_ALLOC_VOID(lib, func) void func (alloc_t buf) { FREE_ALL(dlcallSync(lib, #func, buf, string(-4, -4))); }
#define DLFUNC_CSTR_VOID(lib, func) void func (alloc_t buf) {\
    FREE_ALL(\
        alloc_t buf2 = malloc(strLen(buf+1));\
        strCopy(buf2, buf);\
        setC(buf2, strLen(buf), char(0));\
        dlcallSync(lib, #func, buf2, string(-4, -4));\
    )\
}

DLFUNC_I32_I32_I32(lib, sub)
DLFUNC_CSTR_VOID(lib, hello)

void main() {
    writeSync("hello.c", "#include <stdio.h>\nvoid hello(char* s){printf(\"Hello, %s!\\n\",s);}\nvoid sub(int* v, unsigned int vlen, int* res, unsigned int reslen) { res[0] = v[0]-v[1]; }");
    awaitIO(runCmd("cc --shared -o hello.so hello.c"));

    lib = dlopenSync("./hello.so");

    dlcallSync(lib, "hello", "GLSL\u0000");

    int32_t a = 7, b = 12;
    int32_t subResult = dlcallSync_i32(lib, "sub", a, b);
    println(concat(str(a), " - ", str(b), " = ", str(subResult)));

    hello("GLSL macro");
    a = 8829;
    b = 3741;
    println(concat(str(a), " - ", str(b), " = ", str(sub(a, b))));
}

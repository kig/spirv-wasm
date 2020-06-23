#define version #version
#define extension #extension

version 450
extension GL_EXT_shader_explicit_arithmetic_types : require
extension GL_KHR_memory_scope_semantics : require

#define HEAP_SIZE 8192

struct ioRequest {
    int ioType;
    ivec2 filename;
    int offset;
    int count;
    ivec2 result;
    int status;
};

layout ( local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

layout(std430, binding = 0) buffer inputBuffer { int32_t inputs[]; };
layout(std430, binding = 1) buffer outputBuffer { int32_t outputs[]; };
layout(std430, binding = 2) volatile buffer heapBuffer { int8_t heap[]; };
layout(std430, binding = 3) buffer i32heapBuffer { int32_t i32heap[]; };
layout(std430, binding = 4) volatile buffer ioBuffer { int32_t ioRequests[]; };

#include "file.glsl"

bool testRead() {
    bool okShort = strCmp(readSync("hello.txt", malloc(100)), "Hello, world!") == 0;

    string buf = malloc(100);
    int ok;
    int reqNum = read("hello.txt", 0, 100, buf);
    string res = awaitIO(reqNum, ok);
    bool okLong = strCmp(res, "Hello, world!") == 0;

    return okShort && okLong;
}

bool testWrite() {
    string buf = malloc(100);
    string filename = "write.txt";

    awaitIO(write(filename, 0, 100, "Hello, write!"));
    bool firstOk = strCmp(awaitIO(read(filename, 0, 100, buf)), "Hello, write!") == 0;

    writeSync(filename, "Hello, world!");
    bool secondOk = strCmp(readSync(filename, buf), "Hello, world!") == 0;

    return firstOk && secondOk;
}

void printTest(bool ok, string name) {
    print(name);
    println(ok ? " successful" : " failed!");
}

#define TEST(testFn) printTest(testFn(), #testFn)

void main() {
    initGlobals();

    if (strCmp(readSync("hello.txt", malloc(100)), "Hello, world!") == 0) {
        println("Read test successful!");
    }

    TEST(testRead);
    TEST(testWrite);
}

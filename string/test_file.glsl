#define version #version
version 450

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

layout(std430, binding = 0) readonly buffer inputBuffer { highp int inputs[]; };
layout(std430, binding = 1) buffer outputBuffer { highp int outputs[]; };
layout(std430, binding = 2) buffer heapBuffer { lowp int heap[]; };
layout(std430, binding = 3) buffer ioBuffer { highp int ioRequestsCount; ioRequest ioRequests[]; };

#include "file.glsl"

bool testRead() {
    string buf = malloc(100);
    int ok;
    int reqNum = read("hello.txt", 0, 100, buf);
    string res = awaitIO(reqNum, ok);
    return strCmp(res, "Hello, world!") == 0;
}

bool testWrite() {
    string buf = malloc(100);
    string filename = "write.txt";
    string res;
    int ok, reqNum;

    reqNum = write(filename, 0, 100, "Hello, write!");
    awaitIO(reqNum, ok);
    reqNum = read(filename, 0, 100, buf);
    res = awaitIO(reqNum, ok);
    bool firstOk = strCmp(res, "Hello, write!") == 0;

    reqNum = write(filename, 0, 100, "Hello, world!");
    awaitIO(reqNum, ok);
    reqNum = read(filename, 0, 100, buf);
    res = awaitIO(reqNum, ok);
    bool secondOk = strCmp(res, "Hello, world!") == 0;

    return firstOk && secondOk;
}

void main() {
	initGlobals();
	int heapTop = heapPtr;
	
	int op = int(gl_GlobalInvocationID.x) * 1024;

    outputs[op++] = testRead() ? 1 : -1;
    outputs[op++] = testWrite() ? 1 : -1;
}

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

layout ( local_size_x = 16, local_size_y = 1, local_size_z = 1 ) in;

layout(std430, binding = 0) buffer inputBuffer { int32_t inputs[]; };
layout(std430, binding = 1) buffer outputBuffer { int32_t outputs[]; };
layout(std430, binding = 2) volatile buffer heapBuffer { int8_t heap[]; };
layout(std430, binding = 3) buffer i32heapBuffer { int32_t i32heap[]; };
layout(std430, binding = 4) volatile buffer ioBuffer { int32_t ioRequests[]; };

#include "string.glsl"

#define IO_READ 1
#define IO_WRITE 2
#define IO_CREATE 3
#define IO_DELETE 4
#define IO_TRUNCATE 5

#define IO_NONE 0
#define IO_START 1
#define IO_RECEIVED 2
#define IO_IN_PROGRESS 3
#define IO_COMPLETE 4
#define IO_ERROR 5
#define IO_HANDLED 255

const string stdin = string(0, 0);
const string stdout = string(1, 0);
const string stderr = string(2, 0);

int errno = 0;

int requestIO(ioRequest req) {
    int32_t reqNum = atomicAdd(ioRequests[0], 1);
    int32_t off = 8 + reqNum * 8;
    ioRequests[off+0] = req.ioType;
    ioRequests[off+1] = req.status;
    ioRequests[off+2] = req.offset;
    ioRequests[off+3] = req.count;
    ioRequests[off+4] = req.filename.x;
    ioRequests[off+5] = req.filename.y;
    ioRequests[off+6] = req.result.x;
    ioRequests[off+7] = req.result.y;
    return reqNum;
}

string awaitIO(int reqNum, inout int status) {
    if (ioRequests[8 + reqNum * 8 + 1] != IO_NONE) {
        while (ioRequests[8 + reqNum * 8 + 1] < IO_COMPLETE);
        status = ioRequests[8 + reqNum * 8 + 1];
    }
    ioRequests[8 + reqNum * 8 + 1] = IO_HANDLED;
    string s = string(ioRequests[8 + reqNum * 8 + 6], ioRequests[8 + reqNum * 8 + 7]);
    //memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease);
    return s;
}

string awaitIO(int reqNum) {
    return awaitIO(reqNum, errno);
}

int read(string filename, int offset, int count, string buf) {
    return requestIO(ioRequest(IO_READ, filename, offset, min(count, strLen(buf)), buf, IO_START));
}

int read(string filename, string buf) {
    return read(filename, 0, strLen(buf), buf);
}

int write(string filename, int offset, int count, string buf) {
    return requestIO(ioRequest(IO_WRITE, filename, offset, min(count, strLen(buf)), buf, IO_START));
}

int write(string filename, string buf) {
    return write(filename, 0, strLen(buf), buf);
}

int truncateFile(string filename, int count) {
    return requestIO(ioRequest(IO_TRUNCATE, filename, 0, count, string(0,0), IO_START));
}

int deleteFile(string filename) {
    return requestIO(ioRequest(IO_DELETE, filename, 0, 0, string(0,0), IO_START));
}

int createFile(string filename) {
    return requestIO(ioRequest(IO_CREATE, filename, 0, 0, string(0,0), IO_START));
}

string readSync(string filename, int offset, int count, string buf) { return awaitIO(read(filename, offset, count, buf)); }
string readSync(string filename, string buf) { return awaitIO(read(filename, buf)); }

string writeSync(string filename, int offset, int count, string buf) { return awaitIO(write(filename, offset, count, buf)); }
string writeSync(string filename, string buf) { return awaitIO(write(filename, buf)); }

void print(string message) {
    awaitIO(write(stdout, -1, strLen(message), message));
}

void println(string message) {
    print(concat(message, "\n"));
}

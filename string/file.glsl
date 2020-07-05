#include "string.glsl"

#ifndef TO_CPU_SIZE
#define TO_CPU_SIZE 4096
#endif

#ifndef FROM_CPU_SIZE
#define FROM_CPU_SIZE 4096
#endif

#include "vulkan/io.hpp"

struct ioRequest {
    int32_t ioType;
    int32_t status;
    int64_t offset;
    int64_t count;
    i32vec2 filename;
    i32vec2 data;
    int32_t _pad10;
    int32_t _pad11;
    int32_t _pad12;
    int32_t _pad13;
    int32_t _pad14;
    int32_t _pad15;
};

layout(std430, binding = 1) volatile buffer ioRequestsBuffer {
    int32_t ioCount;
    int32_t programReturnValue;
    int32_t maxIOCount;
    int32_t io_pad_3;
    int32_t io_pad_4;
    int32_t io_pad_5;
    int32_t io_pad_6;
    int32_t io_pad_7;
    int32_t io_pad_8;
    int32_t io_pad_9;
    int32_t io_pad_10;
    int32_t io_pad_11;
    int32_t io_pad_12;
    int32_t io_pad_13;
    int32_t io_pad_14;
    int32_t io_pad_15;

    ioRequest ioRequests[]; 
};

layout(std430, binding = 2) volatile buffer fromCPUBuffer { char fromCPU[]; };
layout(std430, binding = 2) volatile buffer i64v4fromCPUBuffer { i64vec4 i64v4fromCPU[]; };
layout(std430, binding = 3) buffer toCPUBuffer { char toCPU[]; };
layout(std430, binding = 3) buffer i64v4toCPUBuffer { i64vec4 i64v4toCPU[]; };

#define FREE_IO(f) { ptr_t _ihp_ = fromCPUPtr, _thp_ = toCPUPtr; f; fromCPUPtr = _ihp_; toCPUPtr = _thp_; }

const string stdin = string(0, 0);
const string stdout = string(1, 0);
const string stderr = string(2, 0);

int32_t maxIOCountMinusOne = maxIOCount - 1;

int32_t errno = 0;

ptr_t fromCPUStart = ThreadID * FROM_CPU_SIZE;
ptr_t fromCPUEnd = heapStart + FROM_CPU_SIZE;

ptr_t fromCPUPtr = fromCPUStart;

ptr_t toCPUStart = ThreadID * TO_CPU_SIZE;
ptr_t toCPUEnd = heapStart + TO_CPU_SIZE;

ptr_t toCPUPtr = toCPUStart;

stringArray argv = stringArray(
    i32heap[ThreadCount * HEAP_SIZE/4],
    i32heap[ThreadCount * HEAP_SIZE/4 + 1]
);

struct io {
    int32_t index;
    ptr_t heapBufStart;
};

alloc_t toCPUMalloc(size_t len) {
    ptr_t ptr = ((toCPUPtr+31) / 32) * 32;
    toCPUPtr = ptr + ((len+31) / 32) * 32;
    return string(ptr, ptr + len);
}

alloc_t fromCPUMalloc(size_t len) {
    ptr_t ptr = ((fromCPUPtr+31) / 32) * 32;
    fromCPUPtr = ptr + ((len+31) / 32) * 32;
    return string(ptr, ptr + len);
}

void setReturnValue(int32_t i) {
    programReturnValue = i;
}

#define COPYFUNC(NAME, SRC, DST) alloc_t NAME(alloc_t src, alloc_t dst) {\
    ptr_t s=src.x, d=dst.x;\
    if ((d & 31) == 0 && (s & 31) == 0) {\
        for (; d<dst.y - 31 && s<src.y; d+=32, s+=32) {\
            i64v4##DST[d/32] = i64v4##SRC[s/32];\
        }\
    }\
    for (;s<src.y && d<dst.y; s++, d++) {\
        DST[d] = SRC[s];\
    }\
    return dst;\
}

COPYFUNC(copyHeapToCPU, heap, toCPU)

COPYFUNC(copyFromCPUToHeap, fromCPU, heap)

io requestIO(ioRequest request) {
    io token = io(0, request.data.x);
    if (strLen(request.filename) > 0) {
        request.filename = copyHeapToCPU(request.filename, toCPUMalloc(strLen(request.filename)));
    }
    if (request.count > 0 && (request.ioType == IO_READ || request.ioType == IO_LS || request.ioType == IO_GETCWD)) {
        request.data = fromCPUMalloc(size_t(request.count));
        request.data.y = -1;
    } else if (request.count > 0 && request.ioType == IO_WRITE) {
        request.data = copyHeapToCPU(request.data, toCPUMalloc(size_t(request.count)));
        request.data.y = -1;
    }
    memoryBarrier(); //memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
    // Replace this with a proper ring buffer that doesn't have issues with wrapping ioCounts.
    int32_t reqNum = atomicAdd(ioCount, 1) & maxIOCountMinusOne;
    ioRequests[reqNum] = request;
    token.index = reqNum;
    return token;
}

alloc_t awaitIO(io ioReq, inout int32_t status, bool noCopy) {
    if (ioRequests[ioReq.index].status != IO_NONE) {
        while (
            ioRequests[ioReq.index].status < IO_COMPLETE ||
            ioRequests[ioReq.index].data.y == -1
        );
    }

    ioRequest req = ioRequests[ioReq.index];
    status = req.status;
    memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
    if (req.ioType == IO_LS) {
        stringArray res = stringArray(
            toIndexPtr(ioReq.heapBufStart),
            toIndexPtr(ioReq.heapBufStart) + req.offset*2
        );
        FREE(
            heapPtr = fromIndexPtr(res.y);
            ptr_t p = req.data.x;
            for (int i = 0; i < req.offset; i++) {
                int32_t len = int32_t(fromCPU[p]) 
                            | (int32_t(fromCPU[p+1]) << 8) 
                            | (int32_t(fromCPU[p+2]) << 16) 
                            | (int32_t(fromCPU[p+3]) << 24)
                            ;
                p += 4;
                string s = malloc(len);
                copyFromCPUToHeap(string(p, p+len), s);
                aSet(res, i, s);
                p += len;
            }
        )
        return res;
    }
    if (noCopy || !(req.ioType == IO_READ || req.ioType == IO_GETCWD)) {
        ioRequests[ioReq.index].status = IO_HANDLED;
        return req.data;
    }

    alloc_t s = alloc_t(ioReq.heapBufStart, ioReq.heapBufStart + strLen(req.data));
    copyFromCPUToHeap(req.data, s);
    ioRequests[ioReq.index].status = IO_HANDLED;

    return s;
}

alloc_t awaitIO(io request, inout int32_t status) {
    return awaitIO(request, status, false);
}

alloc_t awaitIO(io request) {
    return awaitIO(request, errno);
}

alloc_t awaitIO(io request, bool noCopy) {
    return awaitIO(request, errno, noCopy);
}

io read(string filename, int64_t offset, size_t count, string buf) {
    return requestIO(ioRequest(IO_READ, IO_START, offset, min(count, strLen(buf)), filename, buf,0,0,0,0,0,0));
}

io read(string filename, string buf) {
    return read(filename, 0, strLen(buf), buf);
}

io write(string filename, int64_t offset, size_t count, string buf) {
    return requestIO(ioRequest(IO_WRITE, IO_START, offset, min(count, strLen(buf)), filename, buf,0,0,0,0,0,0));
}

io write(string filename, string buf) {
    return write(filename, 0, strLen(buf), buf);
}

io truncateFile(string filename, size_t count) {
    return requestIO(ioRequest(IO_TRUNCATE, IO_START, 0, count, filename, string(0,0),0,0,0,0,0,0));
}

io deleteFile(string filename) {
    return requestIO(ioRequest(IO_DELETE, IO_START, 0, 0, filename, string(0,0),0,0,0,0,0,0));
}

io createFile(string filename) {
    return requestIO(ioRequest(IO_CREATE, IO_START, 0, 0, filename, string(0,0),0,0,0,0,0,0));
}

io runCmd(string cmd) {
    return requestIO(ioRequest(IO_RUN_CMD, IO_START, 0, 0, cmd, string(0,0),0,0,0,0,0,0));
}

io mkdir(string dirname) {
    return requestIO(ioRequest(IO_MKDIR, IO_START, 0, 0, dirname, string(0,0),0,0,0,0,0,0));
}

io rmdir(string dirname) {
    return requestIO(ioRequest(IO_RMDIR, IO_START, 0, 0, dirname, string(0,0),0,0,0,0,0,0));
}

io ls(string dirname, alloc_t data) {
    return requestIO(ioRequest(IO_LS, IO_START, 0, strLen(data), dirname, data,0,0,0,0,0,0));
}

io ls(string dirname, alloc_t data, int32_t offset) {
    return requestIO(ioRequest(IO_LS, IO_START, offset, strLen(data), dirname, data,0,0,0,0,0,0));
}

io getCwd(string data) {
    return requestIO(ioRequest(IO_GETCWD, IO_START, 0, strLen(data), string(0,0), data,0,0,0,0,0,0));
}

io getCwd() {
    return requestIO(ioRequest(IO_GETCWD, IO_START, 0, 256, string(0,0), malloc(256),0,0,0,0,0,0));
}

io chdir(string dirname) {
    return requestIO(ioRequest(IO_CD, IO_START, 0, 0, dirname, string(0,0),0,0,0,0,0,0));
}

io stat(string filename, alloc_t st) {
    return requestIO(ioRequest(IO_STAT, IO_START, 0, 0, filename, st,0,0,0,0,0,0));
}

io stat(string filename) {
    return requestIO(ioRequest(IO_STAT, IO_START, 0, 0, filename, string(0,0),0,0,0,0,0,0));
}

io open(string filename) {
    return requestIO(ioRequest(IO_OPEN, IO_START, 0, 0, filename, string(0,0),0,0,0,0,0,0));
}

io close(int32_t fd) {
    return requestIO(ioRequest(IO_CLOSE, IO_START, 0, 0, string(fd, 0), string(0,0),0,0,0,0,0,0));
}




alloc_t readSync(string filename, int64_t offset, size_t count, string buf) { return awaitIO(read(filename, offset, count, buf)); }
alloc_t readSync(string filename, string buf) { return awaitIO(read(filename, buf)); }

alloc_t writeSync(string filename, int64_t offset, size_t count, string buf) { return awaitIO(write(filename, offset, count, buf)); }
alloc_t writeSync(string filename, string buf) { return awaitIO(write(filename, buf)); }

void print(string message) {
    FREE_IO(awaitIO(write(stdout, -1, strLen(message), message)));
}

void println(string message) {
    FREE(print(concat(message, str('\n'))));
}

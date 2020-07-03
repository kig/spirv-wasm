#include "string.glsl"

struct ioRequest {
    int32_t ioType;
    int32_t status;
    int64_t offset;
    int64_t count;
    i32vec2 filename;
    i32vec2 data;
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

    ioRequest ioRequests[]; 
};

layout(std430, binding = 2) volatile buffer ioHeapBuffer { char ioHeap[]; };
layout(std430, binding = 2) volatile buffer i64v4IOBuffer { i64vec4 i64v4IOHeap[]; };

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

#ifndef IO_HEAP_SIZE
#define IO_HEAP_SIZE 4096
#endif

#define FREE_IO(f) { ptr_t _ihp_ = ioHeapPtr; f; ioHeapPtr = _ihp_; }

const string stdin = string(0, 0);
const string stdout = string(1, 0);
const string stderr = string(2, 0);

int32_t errno = 0;

ptr_t ioHeapStart = ThreadID * IO_HEAP_SIZE;
ptr_t ioHeapEnd = heapStart + IO_HEAP_SIZE;

ptr_t ioHeapPtr = ioHeapStart;

stringArray argv = stringArray(
    i32heap[ThreadCount * HEAP_SIZE/4],
    i32heap[ThreadCount * HEAP_SIZE/4 + 1]
);

struct io {
    int32_t index;
    ptr_t heapBufStart;
};

alloc_t ioMalloc(size_t len) {
    ptr_t ptr = ((ioHeapPtr+31) / 32) * 32;
    ioHeapPtr = ptr + ((len+31) / 32) * 32;
    return string(ptr, ptr + len);
}

void setReturnValue(int32_t i) {
    programReturnValue = i;
}

io requestIO(ioRequest request) {
    io token = io(0, request.data.x);
    if (request.count > 0 && request.ioType == IO_READ) {
        string b = ioMalloc(size_t(request.count));
        request.data = b;
    }
    //memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
    // Replace this with a proper ring buffer that doesn't have issues with wrapping ioCounts.
    int32_t reqNum = atomicAdd(ioCount, 1); // % maxIOCount;

    ioRequests[reqNum] = request;
    token.index = reqNum;
    return token;
}

alloc_t awaitIO(io ioReq, inout int32_t status, bool noCopy) {
    if (ioRequests[ioReq.index].status != IO_NONE && ioRequests[ioReq.index].status < IO_COMPLETE) {
        while (ioRequests[ioReq.index].status < IO_COMPLETE);
    }

    ioRequest req = ioRequests[ioReq.index];
    status = req.status;

    //memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
    if (noCopy || req.ioType != IO_READ) {
        ioRequests[ioReq.index].status = IO_HANDLED;
        return req.data;
    }

    alloc_t s = alloc_t(ioReq.heapBufStart, ioReq.heapBufStart + strLen(req.data));

    ptr_t i = s.x, x = req.data.x, y = req.data.y;

    if ((i % 32) == 0) {
        for (; i < s.y - 31 && x < y; x+=32, i+=32) {
            i64v4heap[i/32] = i64v4IOHeap[x/32];
        }
    }
    for (; i < s.y && x < req.data.y; x++, i++) {
        heap[i] = ioHeap[x];
    }
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
    return requestIO(ioRequest(IO_READ, IO_START, offset, min(count, strLen(buf)), filename, buf));
}

io read(string filename, string buf) {
    return read(filename, 0, strLen(buf), buf);
}

io write(string filename, int64_t offset, size_t count, string buf) {
    return requestIO(ioRequest(IO_WRITE, IO_START, offset, min(count, strLen(buf)), filename, buf));
}

io write(string filename, string buf) {
    return write(filename, 0, strLen(buf), buf);
}

io truncateFile(string filename, size_t count) {
    return requestIO(ioRequest(IO_TRUNCATE, IO_START, 0, count, filename, string(0,0)));
}

io deleteFile(string filename) {
    return requestIO(ioRequest(IO_DELETE, IO_START, 0, 0, filename, string(0,0)));
}

io createFile(string filename) {
    return requestIO(ioRequest(IO_CREATE, IO_START, 0, 0, filename, string(0,0)));
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

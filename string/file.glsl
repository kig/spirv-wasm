#include <string.glsl>
#include <io.glsl>
#include <errno.glsl>

struct ioRequest {
    int32_t ioType;
    int32_t status;
    int64_t offset;
    int64_t count;
    i32vec2 filename;
    i32vec2 data;
    int32_t compression;
    int32_t progress;
    i32vec2 data2;
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

layout(std430, binding = 2) volatile buffer fromIOBuffer { char fromIO[]; };
layout(std430, binding = 2) volatile buffer u8fromIOBuffer { uint8_t u8fromIO[]; };
layout(std430, binding = 2) volatile buffer i16fromIOBuffer { int16_t i16fromIO[]; }; // 2 bytes
layout(std430, binding = 2) volatile buffer i32fromIOBuffer { int32_t i32fromIO[]; }; // 4 bytes
layout(std430, binding = 2) volatile buffer i64fromIOBuffer { int64_t i64fromIO[]; }; // 8 bytes
layout(std430, binding = 2) volatile buffer i64v2fromIOBuffer { i64vec2 i64v2fromIO[]; }; // 16 bytes
layout(std430, binding = 2) volatile buffer i64v4fromIOBuffer { i64vec4 i64v4fromIO[]; }; // 32 bytes
layout(std430, binding = 2) volatile buffer f64m42fromIOBuffer { f64mat4x2 f64m42fromIO[]; }; // 64 bytes
layout(std430, binding = 2) volatile buffer f64m4fromIOBuffer { f64mat4 f64m4fromIO[]; }; // 128 bytes
layout(std430, binding = 3) buffer toIOBuffer { char toIO[]; };
layout(std430, binding = 3) buffer i64v4toIOBuffer { i64vec4 i64v4toIO[]; };

#define FREE_IO(f) { ptr_t _ihp_ = fromIOPtr, _thp_ = toIOPtr; f; fromIOPtr = _ihp_; toIOPtr = _thp_; }

const string stdin = string(0, 0);
const string stdout = string(1, 0);
const string stderr = string(2, 0);

int32_t errno = 0;

ptr_t fromIOStart = ThreadId * FromIOSize;
ptr_t fromIOEnd = fromIOStart + FromIOSize;

ptr_t fromIOPtr = fromIOStart;

ptr_t toIOStart = ThreadId * ToIOSize;
ptr_t toIOEnd = toIOStart + ToIOSize;

ptr_t toIOPtr = toIOStart;

ptr_t groupFromIOStart = ThreadGroupId * GroupFromIOSize;
ptr_t groupFromIOEnd = groupFromIOStart + GroupFromIOSize;
shared ptr_t groupFromIOPtr;

ptr_t groupToIOStart = ThreadGroupId * GroupToIOSize;
ptr_t groupToIOEnd = groupToIOStart + GroupToIOSize;
shared ptr_t groupToIOPtr;

#include <binary_data.glsl>
#include <stat.glsl>

stringArray argv = stringArray(
    i32heap[HeapGlobalsOffset/4 - 2],
    i32heap[HeapGlobalsOffset/4 - 1]
);

struct io {
    int32_t index;
    ptr_t heapBufStart;
};

alloc_t toIOMalloc(size_t len) {
    ptr_t ptr = ((toIOPtr+31) / 32) * 32;
    toIOPtr = ptr + ((len+31) / 32) * 32;
    return string(ptr, ptr + len);
}

alloc_t fromIOMalloc(size_t len) {
    ptr_t ptr = ((fromIOPtr+31) / 32) * 32;
    fromIOPtr = ptr + ((len+31) / 32) * 32;
    return string(ptr, ptr + len);
}

void setReturnValue(int32_t i) {
    programReturnValue = i;
}

int32_t maxIOCount_cached = maxIOCount;

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

COPYFUNC(copyHeapToIO, heap, toIO)

COPYFUNC(copyFromIOToHeap, fromIO, heap)

io requestIO(ioRequest request) {
    io token = io(0, request.data.x);
    memoryBarrier();
    // memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
    if (strLen(request.filename) > 0) {
        request.filename = copyHeapToIO(request.filename, toIOMalloc(strLen(request.filename)));
    }
    if (request.count > 0 && (request.ioType == IO_READ || request.ioType == IO_LS || request.ioType == IO_GETCWD || request.ioType == IO_DLOPEN)) {
        request.data = fromIOMalloc(size_t(strLen(request.data)));
        request.data.y = -1;
    } else if (request.count > 0 && (request.ioType == IO_WRITE || request.ioType == IO_COPY)) {
        request.data = copyHeapToIO(request.data, toIOMalloc(size_t(request.count)));
        request.data.y = -1;
    }
    if (request.count > 0 && (request.ioType == IO_DLCALL)) {
        request.data = copyHeapToIO(request.data, toIOMalloc(size_t(request.count)));
        request.data.y = -1;
        request.data2 = fromIOMalloc(size_t(strLen(request.data2)));
    }
    memoryBarrier();
    // memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);

    // Replace this with a proper ring buffer that doesn't have issues with wrapping ioCounts.
    int32_t reqNum = atomicAdd(ioCount, 1) % maxIOCount_cached;

    // Wait for possible previous IO to complete :<
    while(ioRequests[reqNum].status != IO_NONE && ioRequests[reqNum].status < IO_COMPLETE);
    
    ioRequests[reqNum] = request;
    token.index = reqNum;
    return token;
}

alloc_t awaitIO(io ioReq, inout int32_t status, bool noCopy, out size_t ioCount, out bool compressed) {
    if (ioRequests[ioReq.index].status != IO_NONE) {
        while (ioRequests[ioReq.index].status < IO_COMPLETE ||
               ioRequests[ioReq.index].data.y == -1);
    }

    ioRequest req = ioRequests[ioReq.index];
    status = req.status;
    compressed = (req.compression != 0);
    ioCount = size_t(req.count);

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
                int32_t len = int32_t(fromIO[p]) 
                            | (int32_t(fromIO[p+1]) << 8) 
                            | (int32_t(fromIO[p+2]) << 16) 
                            | (int32_t(fromIO[p+3]) << 24)
                            ;
                p += 4;
                string s = malloc(len);
                copyFromIOToHeap(string(p, p+len), s);
                aSet(res, i, s);
                p += len;
            }
        )
        ioRequests[ioReq.index].status = IO_HANDLED;
        return res;
    } else if (req.ioType == IO_DLCALL) {
        if (noCopy) return req.data2;
        alloc_t s = alloc_t(ioReq.heapBufStart, ioReq.heapBufStart + strLen(req.data2));
        copyFromIOToHeap(req.data2, s);
        ioRequests[ioReq.index].status = IO_HANDLED;
        return s;
    }

    alloc_t s;
    // Leave data in the IO buffer if the noCopy flag is set or if the IO is not one that returns data.
    if (noCopy || !(req.ioType == IO_READ || req.ioType == IO_GETCWD || req.ioType == IO_STAT || req.ioType == IO_DLOPEN)) {
        s = req.data;
    } else {
        s = alloc_t(ioReq.heapBufStart, ioReq.heapBufStart + strLen(req.data));
        copyFromIOToHeap(req.data, s);
    }
    ioRequests[ioReq.index].status = IO_HANDLED;

    return s;
}

alloc_t awaitIO(io request, inout int32_t status) {
    size_t count;
    bool compressed;
    return awaitIO(request, status, false, count, compressed);
}

alloc_t awaitIO(io request) {
    size_t count;
    bool compressed;
    return awaitIO(request, errno, false, count, compressed);
}

alloc_t awaitIO(io request, bool noCopy) {
    size_t count;
    bool compressed;
    return awaitIO(request, errno, noCopy, count, compressed);
}

alloc_t awaitIO(io request, bool noCopy, out size_t count) {
    bool compressed;
    return awaitIO(request, errno, noCopy, count, compressed);
}

alloc_t awaitIO(io request, bool noCopy, out size_t count, out bool compressed) {
    return awaitIO(request, errno, noCopy, count, compressed);
}

io read(string filename, int64_t offset, size_t count, string buf) {
    return requestIO(ioRequest(IO_READ, IO_START, offset, min(count, strLen(buf)), filename, buf,0,0,string(0,0),0,0));
}

io read(string filename, int64_t offset, size_t count, string buf, int32_t compression) {
    return requestIO(ioRequest(IO_READ, IO_START, offset, min(count, strLen(buf)), filename, buf, compression, 0,string(0,0),0,0));
}

io read(string filename, string buf) {
    return read(filename, 0, strLen(buf), buf);
}

io write(string filename, int64_t offset, size_t count, string buf) {
    return requestIO(ioRequest(IO_WRITE, IO_START, offset, min(count, strLen(buf)), filename, buf,0,0,string(0,0),0,0));
}

io write(string filename, string buf) {
    return write(filename, 0, strLen(buf), buf);
}

io truncateFile(string filename, size_t count) {
    return requestIO(ioRequest(IO_TRUNCATE, IO_START, 0, count, filename, string(0,0),0,0,string(0,0),0,0));
}

io deleteFile(string filename) {
    return requestIO(ioRequest(IO_DELETE, IO_START, 0, 0, filename, string(0,0),0,0,string(0,0),0,0));
}

io createFile(string filename) {
    return requestIO(ioRequest(IO_CREATE, IO_START, 0, 0, filename, string(0,0),0,0,string(0,0),0,0));
}

io runCmd(string cmd) {
    return requestIO(ioRequest(IO_RUN_CMD, IO_START, 0, 0, cmd, string(0,0),0,0,string(0,0),0,0));
}

io mkdir(string dirname) {
    return requestIO(ioRequest(IO_MKDIR, IO_START, 0, 0, dirname, string(0,0),0,0,string(0,0),0,0));
}

io rmdir(string dirname) {
    return requestIO(ioRequest(IO_RMDIR, IO_START, 0, 0, dirname, string(0,0),0,0,string(0,0),0,0));
}

io ls(string dirname, alloc_t data) {
    return requestIO(ioRequest(IO_LS, IO_START, 0, strLen(data), dirname, data,0,0,string(0,0),0,0));
}

io ls(string dirname, alloc_t data, int32_t offset) {
    return requestIO(ioRequest(IO_LS, IO_START, offset, strLen(data), dirname, data,0,0,string(0,0),0,0));
}

io getCwd(string data) {
    return requestIO(ioRequest(IO_GETCWD, IO_START, 0, strLen(data), string(0,0), data,0,0,string(0,0),0,0));
}

io getCwd() {
    return requestIO(ioRequest(IO_GETCWD, IO_START, 0, 256, string(0,0), malloc(256),0,0,string(0,0),0,0));
}

io chdir(string dirname) {
    return requestIO(ioRequest(IO_CD, IO_START, 0, 0, dirname, string(0,0), 0,0,string(0,0),0,0));
}

io stat(string filename, alloc_t statBuf) {
    return requestIO(ioRequest(IO_STAT, IO_START, 0, min(strLen(statBuf), StatSize), filename, statBuf, 0,0,string(0,0),0,0));
}

io open(string filename) {
    return requestIO(ioRequest(IO_OPEN, IO_START, 0, 0, filename, string(0,0), 0,0,string(0,0),0,0));
}

io close(int32_t fd) {
    return requestIO(ioRequest(IO_CLOSE, IO_START, 0, 0, string(fd, 0), string(0,0), 0,0,string(0,0),0,0));
}

io exit(int32_t exitCode) {
    return requestIO(ioRequest(IO_EXIT, IO_START, exitCode, 0, string(0,0), string(0,0), 0,0,string(0,0),0,0));
}

alloc_t readSync(string filename, int64_t offset, size_t count) { return awaitIO(read(filename, offset, count, malloc(count))); }
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

void eprint(string message) {
    FREE_IO(awaitIO(write(stderr, -1, strLen(message), message)));
}

void eprintln(string message) {
    FREE(eprint(concat(message, str('\n'))));
}

void log(string message) {
    if (ThreadLocalId == 0) eprintln(message);
}

Stat awaitStat(io request) {
    return initStat(awaitIO(request));
}

Stat statSync(string filename) {
    Stat st;
    FREE_IO(FREE(
        st = awaitStat(stat(filename, malloc(StatSize)));
    ))
    return st;
}

bool fileExistsSync(string filename) {
    return statSync(filename).error == 0;
}


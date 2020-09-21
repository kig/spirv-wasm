#include <string.glsl>
#include <io.glsl>
#include <errno.glsl>

struct ioRequest {
    int32_t ioType;
    int32_t status;
    int64_t offset;
    int64_t count;
    alloc_t filename;
    alloc_t data;
    int32_t compression;
    int32_t progress;
    alloc_t data2;
    int32_t _pad14;
    int32_t _pad15;
};

int stopIO = 0;

layout(std430, binding = 1) volatile buffer ioRequestsBuffer {
    int32_t ioCount;
    int32_t programReturnValue;
    int32_t maxIOCount;
    int32_t runCount;
    int32_t rerunProgram;
    int32_t exited;
    int32_t io_pad_6;
    int32_t io_pad_7;
    int32_t io_pad_8;
    int32_t io_pad_9;
    int32_t io_pad_10;
    int32_t io_pad_11;
    int64_t io_pad_12;
    //int32_t io_pad_13;
    int64_t io_pad_14;
    //int32_t io_pad_15;

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
layout(std430, binding = 3) buffer u8toIOBuffer { uint8_t u8toIO[]; };
layout(std430, binding = 3) buffer i16toIOBuffer { int16_t i16toIO[]; };
layout(std430, binding = 3) buffer i32toIOBuffer { int32_t i32toIO[]; };
layout(std430, binding = 3) buffer i64toIOBuffer { int64_t i64toIO[]; };
layout(std430, binding = 3) buffer i64v2toIOBuffer { i64vec2 i64v2toIO[]; };
layout(std430, binding = 3) buffer i64v4toIOBuffer { i64vec4 i64v4toIO[]; };
layout(std430, binding = 3) buffer f64m42toIOBuffer { f64mat4x2 f64m42toIO[]; };
layout(std430, binding = 3) buffer f64m4toIOBuffer { f64mat4 f64m4toIO[]; };

#define FREE_IO(f) { ptr_t _ihp_ = fromIOPtr, _thp_ = toIOPtr; f; fromIOPtr = _ihp_; toIOPtr = _thp_; }
#define FREE_ALL(f) FREE(FREE_IO(f))

#define socket string
#define file string

const string stdin = string(0, 0);
const string stdout = string(1, 0);
const string stderr = string(2, 0);

int32_t ioError = 0;

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
    if ((d & 127) == 0 && (s & 127) == 0) {\
        for (; d<dst.y - 127 && s<src.y - 127; d+=128, s+=128) {\
            f64m4##DST[d/128] = f64m4##SRC[s/128];\
        }\
    }\
    if ((d & 31) == 0 && (s & 31) == 0) {\
        for (; d<dst.y - 31 && s<src.y - 31; d+=32, s+=32) {\
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

bool isReadIOType(int32_t ioType) {
    return (ioType == IO_READ || ioType == IO_READLINE || ioType == IO_LS || ioType == IO_GETCWD || ioType == IO_STAT || ioType == IO_DLOPEN || ioType == IO_RECV || ioType == IO_MEMREAD || ioType == IO_MALLOC);
}

bool isWriteIOType(int32_t ioType) {
    return (ioType == IO_WRITE || ioType == IO_COPY || ioType == IO_SEND || ioType == IO_MEMWRITE);
}

bool isReadWriteIOType(int32_t ioType) {
    return (ioType == IO_DLCALL || ioType == IO_ACCEPT);
}

io requestIO(ioRequest request, bool needToCopyDataToIO) {
    io token = io(0, request.data.x);
    if ((exited != 0 || stopIO != 0)) return token;
    memoryBarrier();
    // memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
    if (strLen(request.filename) > 0) {
        request.filename = copyHeapToIO(request.filename, toIOMalloc(strLen(request.filename)));
    }
    if (request.count > 0 && isReadIOType(request.ioType)) {
        request.data = fromIOMalloc(size_t(strLen(request.data)));
        request.data.y = -1;
    } else if (request.count > 0 && isWriteIOType(request.ioType)) {
        if (needToCopyDataToIO) request.data = copyHeapToIO(request.data, toIOMalloc(size_t(request.count)));
        request.data.y = -1;
    }
    if (request.count > 0 && isReadWriteIOType(request.ioType)) {
        if (needToCopyDataToIO) request.data = copyHeapToIO(request.data, toIOMalloc(size_t(request.count)));
        request.data.y = -1;
        token.heapBufStart = request.data2.x;
        request.data2 = fromIOMalloc(size_t(strLen(request.data2)));
    }
    memoryBarrier();
    // memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);

    // Replace this with a proper ring buffer that doesn't have issues with wrapping ioCounts.
    int32_t reqNum = atomicAdd(ioCount, 1) % maxIOCount_cached;

    // Wait for possible previous IO to complete :<
    // This will deadlock if the IO buffer is full of requests and you're not awaiting for them.
    while(atomicAdd(ioRequests[reqNum].status, 0) != IO_NONE && atomicAdd(ioRequests[reqNum].status, 0) != IO_HANDLED);

    ioRequests[reqNum] = request;
    token.index = reqNum;
    return token;
}

io requestIO(ioRequest request) {
    return requestIO(request, true);
}

bool pollIO(io ioReq) {
    return (
        (exited != 0 || stopIO != 0) || ioRequests[ioReq.index].status >= IO_COMPLETE && ioRequests[ioReq.index].data.y != -1
    );
}

void awaitAndDiscardIO(io ioReq) {
    if ((exited != 0 || stopIO != 0)) return;
    if (atomicAdd(ioRequests[ioReq.index].status, 0) != IO_NONE) {
        while (atomicAdd(ioRequests[ioReq.index].status, 0) < IO_COMPLETE ||
               atomicAdd(ioRequests[ioReq.index].data.y, 0) == -1);
    }
    ioRequests[ioReq.index].status = IO_HANDLED;
}

alloc_t awaitIO(io ioReq, inout int32_t status, bool noCopy, out size_t ioCount, out bool compressed) {
    if ((exited != 0 || stopIO != 0)) return alloc_t(-1,-1);
    if (atomicAdd(ioRequests[ioReq.index].status, 0) != IO_NONE) {
        while (atomicAdd(ioRequests[ioReq.index].status, 0) < IO_COMPLETE ||
               atomicAdd(ioRequests[ioReq.index].data.y, 0) == -1);
    }

    ioRequest req = ioRequests[ioReq.index];
    status = req.status;
    compressed = (req.compression != 0);
    ioCount = size_t(req.count);

    memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);

    if (req.ioType == IO_LS) {
        stringArray res = stringArray(
            toIndexPtr(ioReq.heapBufStart),
            toIndexPtr(ioReq.heapBufStart) + int32_t(req.offset*2)
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
    if (noCopy || !isReadIOType(req.ioType)) {
        s = req.data;
    } else {
        s = alloc_t(ioReq.heapBufStart, ioReq.heapBufStart + strLen(req.data));
        copyFromIOToHeap(req.data, s);
    }
    ioRequests[ioReq.index].status = IO_HANDLED;

    return s;
}

alloc_t awaitIO2(io ioReq, out string data2) {
    if ((exited != 0 || stopIO != 0)) return alloc_t(-1,-1);
    if (ioRequests[ioReq.index].status != IO_NONE) {
        while (ioRequests[ioReq.index].status < IO_COMPLETE ||
               ioRequests[ioReq.index].data.y == -1);
    }

    ioRequest req = ioRequests[ioReq.index];

    memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);

    alloc_t s = alloc_t(ioReq.heapBufStart, ioReq.heapBufStart + strLen(req.data2));
    copyFromIOToHeap(req.data2, s);
    data2 = s;
    ioRequests[ioReq.index].status = IO_HANDLED;

    return req.data;
}

alloc_t awaitIO(io request, inout int32_t status) {
    size_t count;
    bool compressed;
    return awaitIO(request, status, false, count, compressed);
}

alloc_t awaitIO(io request) {
    size_t count;
    bool compressed;
    return awaitIO(request, ioError, false, count, compressed);
}

alloc_t awaitIO(io request, bool noCopy) {
    size_t count;
    bool compressed;
    return awaitIO(request, ioError, noCopy, count, compressed);
}

alloc_t awaitIO(io request, bool noCopy, out size_t count) {
    bool compressed;
    return awaitIO(request, ioError, noCopy, count, compressed);
}

alloc_t awaitIO(io request, bool noCopy, out size_t count, out bool compressed) {
    return awaitIO(request, ioError, noCopy, count, compressed);
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

io readLine(string filename, size_t count, string buf) {
    return requestIO(ioRequest(IO_READLINE, IO_START, 0, min(count, strLen(buf)), filename, buf, 0,0, string(0,0), 0,0));
}

io readLine(string filename, string buf) {
    return readLine(filename, strLen(buf), buf);
}

io write(string filename, int64_t offset, size_t count, string buf, bool needToCopyDataToIO) {
    return requestIO(ioRequest(IO_WRITE, IO_START, offset, min(count, strLen(buf)), filename, buf,0,0,string(0,0),0,0), needToCopyDataToIO);
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

io microTime() {
    return requestIO(ioRequest(IO_TIMENOW, IO_START, 0, 0, string(0,0), string(0,0), 0,0,string(0,0),0,0));
}

int64_t microTimeSync() {
    io r = microTime();
    awaitIO(r);
    return ioRequests[r.index].offset;
}

io open(string filename) {
    return requestIO(ioRequest(IO_OPEN, IO_START, 0, 0, filename, string(0,0), 0,0,string(0,0),0,0));
}

io close(file fd) {
    return requestIO(ioRequest(IO_CLOSE, IO_START, 0, 0, fd, string(0,0), 0,0,string(0,0),0,0));
}

io exit(int32_t exitCode) {
    exited = 1;
    programReturnValue = exitCode;
    return requestIO(ioRequest(IO_EXIT, IO_START, exitCode, 0, string(0,0), string(0,0), 0,0,string(0,0),0,0));
}


io listen(int32_t port) {
    return requestIO(ioRequest(IO_LISTEN, IO_START, port, 0, string(0,0), string(0,0), 0,0,string(0,0),0,0));
}

io accept(socket sock) {
    return requestIO(ioRequest(IO_ACCEPT, IO_START, 0, 0, sock, string(0,0), 0,0,string(0,0),0,0), false);
}

io recv(socket sock, int32_t count, string data) {
    return requestIO(ioRequest(IO_RECV, IO_START, 0, min(count, strLen(data)), sock, data, 0,0,string(0,0),0,0));
}

io acceptAndRecv(socket sock, alloc_t recvBuf) {
    return requestIO(ioRequest(IO_ACCEPT, IO_START, 0, strLen(recvBuf), sock, string(0,0), 0,0, recvBuf, 0,0), false);
}


io send(socket sock, int32_t count, string data) {
    return requestIO(ioRequest(IO_SEND, IO_START, 0, min(count, strLen(data)), sock, data, 0,0,string(0,0),0,0));
}

io sendAndClose(socket sock, string data) {
    return requestIO(ioRequest(IO_SEND, IO_START, 0, strLen(data), sock, data, 0,-1,string(0,0),0,0));
}

io memAlloc(int64_t count, string data) {
    return requestIO(ioRequest(IO_MALLOC, IO_START, 0, count, string(0,0), data, 0,0,string(0,0),0,0));
}

io memFree(int64_t ptr) {
    return requestIO(ioRequest(IO_MEMFREE, IO_START, ptr, 0, string(0,0), string(0,0), 0,0,string(0,0),0,0));
}

io memRead(int64_t ptr, string data) {
    return requestIO(ioRequest(IO_MEMREAD, IO_START, ptr, strLen(data), string(0,0), data, 0,0,string(0,0),0,0));
}

io memWrite(int64_t ptr, string data) {
    return requestIO(ioRequest(IO_MEMWRITE, IO_START, ptr, strLen(data), string(0,0), data, 0,0,string(0,0),0,0));
}

socket listenSync(int32_t port) {
    return awaitIO(listen(port));
}


void exitSync(int32_t code) {
    awaitIO(exit(code));
}

alloc_t readSync(string filename, int64_t offset, size_t count) { return awaitIO(read(filename, offset, count, malloc(count))); }
alloc_t readSync(string filename, int64_t offset, size_t count, string buf) { return awaitIO(read(filename, offset, count, buf)); }
alloc_t readSync(string filename, string buf) { return awaitIO(read(filename, buf)); }

alloc_t writeSync(string filename, int64_t offset, size_t count, string buf) { return awaitIO(write(filename, offset, count, buf)); }
alloc_t writeSync(string filename, string buf) { return awaitIO(write(filename, buf)); }

#define PRINT_STR_2(f) void f(string a, string b) { FREE(f(concat(a, b))); }
#define PRINT_STR_3(f) void f(string a, string b, string c) { FREE(f(concat(a, b, c))); }
#define PRINT_STR_4(f) void f(string a, string b, string c, string d) { FREE(f(concat(a, b, c, d))); }
#define PRINT_STR_5(f) void f(string a, string b, string c, string d, string e) { FREE(f(concat(a, b, c, d, e))); }
#define PRINT_STR_6(f) void f(string a, string b, string c, string d, string e, string g) { FREE(f(concat(a, b, c, d, e, g))); }

#define PRINT_STR_FUNC(f) \
    PRINT_STR_2(f) \
    PRINT_STR_3(f) \
    PRINT_STR_4(f) \
    PRINT_STR_5(f) \
    PRINT_STR_6(f)

void print(string message) {
    FREE_IO(awaitIO(write(stdout, -1, strLen(message), message)));
}

void println_(string message) {
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

PRINT_STR_FUNC(print)
PRINT_STR_FUNC(println_)
PRINT_STR_FUNC(eprint)
PRINT_STR_FUNC(eprintln)
PRINT_STR_FUNC(log)

#define println1_(a) println_(str(a))
#define println2_(a,b) println_(str(a),str(b))
#define println3_(a,b,c) println_(str(a),str(b),str(c))
#define println4_(a,b,c,d) println_(str(a),str(b),str(c),str(d))
#define println5_(a,b,c,d,e) println_(str(a),str(b),str(c),str(d),str(e))
#define println6_(a,b,c,d,e,f) println_(str(a),str(b),str(c),str(d),str(e),str(f))

#define GET_MACRO(_1,_2,_3,_4,_5,_6,NAME,...) NAME
#define println(...) GET_MACRO(__VA_ARGS__, println6_, println5_, println4_, println3_, println2_, println1_)(__VA_ARGS__)


Stat awaitStat(io request) {
    return initStat(awaitIO(request));
}

Stat statSync(string filename) {
    Stat st;
    FREE_ALL(
        st = awaitStat(stat(filename, malloc(StatSize)));
    )
    return st;
}

bool fileExistsSync(string filename) {
    return statSync(filename).error == 0;
}

uint64_t fileSizeSync(string filename) {
    return statSync(filename).st_size;
}

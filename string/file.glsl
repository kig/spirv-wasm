#include "string.glsl"

struct ioRequest {
    int32_t ioType;
    int32_t status;
    int32_t offset;
    int32_t count;
    i32vec2 filename;
    i32vec2 data;
};

layout(std430, binding = 2) volatile buffer ioBuffer { ioRequest ioRequests[]; };
layout(std430, binding = 3) volatile buffer ioBytes { char ioHeap[]; };

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

#define FREE_IO(f) { int _ihp_ = ioHeapPtr; f; ioHeapPtr = _ihp_; }

const string stdin = string(0, 0);
const string stdout = string(1, 0);
const string stderr = string(2, 0);

int errno = 0;

int ioHeapStart = int(gl_GlobalInvocationID.x) * HEAP_SIZE;
int ioHeapEnd = heapStart + HEAP_SIZE;

int ioHeapPtr = ioHeapStart;

struct io {
    int index;
    int heapBufStart;
};

string ioMalloc(int len) {
    int ptr = ioHeapPtr;
    ioHeapPtr += len;
    return string(ptr, ioHeapPtr);
}

string copyHeapToIO(string s, string b) {
    for (int i = s.x, x = b.x; x < b.y && i < s.y; x++, i++) {
        ioHeap[x] = heap[i];
    }
    return b;
}

string copyHeapToIO(string s) {
    return copyHeapToIO(s, ioMalloc(strLen(s)));
}

io requestIO(ioRequest req) {
    int32_t reqNum = atomicAdd(ioRequests[0].ioType, 1) + 1;
    io token = io(reqNum, req.data.x);
    if (strLen(req.filename) > 0) {
        req.filename = copyHeapToIO(req.filename);
    }
    if (req.count > 0) {
        string b = ioMalloc(req.count);
        if (req.ioType == IO_WRITE) {
            copyHeapToIO(req.data, b);
        }
        req.data = b;
    }
    ioRequests[reqNum] = req;
    return token;
}

string awaitIO(io reqNum, inout int status) {
    if (ioRequests[reqNum.index].status != IO_NONE) {
        while (ioRequests[reqNum.index].status < IO_COMPLETE);
    }
    ioRequest req = ioRequests[reqNum.index];
    status = req.status;
    ioRequests[reqNum.index].status = IO_HANDLED;
    string s = string(reqNum.heapBufStart, reqNum.heapBufStart + strLen(req.data));
    for (int i = s.x, x = req.data.x, y = req.data.y; x < y; x++, i++) {
        heap[i] = ioHeap[x];
    }
    return s;
}

string awaitIO(io reqNum) {
    return awaitIO(reqNum, errno);
}

io read(string filename, int offset, int count, string buf) {
    return requestIO(ioRequest(IO_READ, IO_START, offset, min(count, strLen(buf)), filename, buf));
}

io read(string filename, string buf) {
    return read(filename, 0, strLen(buf), buf);
}

io write(string filename, int offset, int count, string buf) {
    return requestIO(ioRequest(IO_WRITE, IO_START, offset, min(count, strLen(buf)), filename, buf));
}

io write(string filename, string buf) {
    return write(filename, 0, strLen(buf), buf);
}

io truncateFile(string filename, int count) {
    return requestIO(ioRequest(IO_TRUNCATE, IO_START, 0, count, filename, string(0,0)));
}

io deleteFile(string filename) {
    return requestIO(ioRequest(IO_DELETE, IO_START, 0, 0, filename, string(0,0)));
}

io createFile(string filename) {
    return requestIO(ioRequest(IO_CREATE, IO_START, 0, 0, filename, string(0,0)));
}

string readSync(string filename, int offset, int count, string buf) { return awaitIO(read(filename, offset, count, buf)); }
string readSync(string filename, string buf) { return awaitIO(read(filename, buf)); }

string writeSync(string filename, int offset, int count, string buf) { return awaitIO(write(filename, offset, count, buf)); }
string writeSync(string filename, string buf) { return awaitIO(write(filename, buf)); }

void print(string message) {
    FREE_IO(awaitIO(write(stdout, -1, strLen(message), message)));
}

void println(string message) {
    int hp = heapPtr;
    print(concat(message, "\n"));
    heapPtr = hp;
}

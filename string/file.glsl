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

int requestIO(ioRequest req) {
    int reqNum = atomicAdd(ioRequestsCount, 1);
    ioRequests[reqNum] = req;
    return reqNum;
}

int read(string filename, int offset, int count, string buf) {
    return requestIO(ioRequest(IO_READ, filename, offset, min(count, strLen(buf)), buf, IO_START));
}

int write(string filename, int offset, int count, string buf) {
    return requestIO(ioRequest(IO_WRITE, filename, offset, min(count, strLen(buf)), buf, IO_START));
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

string awaitIO(int reqNum, inout int status) {
    if (ioRequests[reqNum].status != IO_NONE) {
        while (ioRequests[reqNum].status < IO_COMPLETE) {
            // wait for completion...
        }
    }
    ioRequest req = ioRequests[reqNum];
    status = req.status;
    return req.result;
}

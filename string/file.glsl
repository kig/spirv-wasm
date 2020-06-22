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
	int cmd = 0;
    if (ioRequests[8 + reqNum * 8 + 1] != IO_NONE) {
        // for (int i = 0; i < 8192; i++) ioRequests[7]++;
        int32_t s = ioRequests[8 + reqNum * 8 + 1];
        for (int i = 0; i < 8192 && s < IO_COMPLETE; i++) {
            memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer | gl_StorageSemanticsOutput, gl_SemanticsAcquireRelease | gl_SemanticsMakeVisible | gl_SemanticsMakeAvailable);
            s = ioRequests[8 + reqNum * 8 + 1];
        }
        while (s < IO_COMPLETE) s = ioRequests[8 + (reqNum+1024) * 8 + 1];
        status = s;
    }
    ioRequests[8 + reqNum * 8 + 1] = IO_HANDLED;
    string s = string(ioRequests[8 + reqNum * 8 + 6], ioRequests[8 + (reqNum+1024) * 8 + 7]);
    return s;
}

#include <file.glsl>
#include <statemachine.glsl>

HeapSize = 8192;
FromIOSize = 8192;
ToIOSize = 8192;

ThreadGroupCount = 256;
ThreadLocalCount = 1;

const int s_Init = 0;
const int s_Accept = 1;
const int s_WaitingConn = 2;
const int s_Reading = 3;
const int s_Writing = 4;
const int s_Closing = 5;

const int a_Server = 0;
const int a_ConnectionIO = 1;
const int a_Connection = 2;
const int a_ReadIO = 3;
const int a_WriteIO = 4;
const int a_CloseIO = 5;
const int a_HeapStart = 6;

string process(string req) {
    return concat("HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\n\r\nHello from ", str(ThreadId), "\n");
}

#define LOAD(k) atomicLoad(k, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire)
#define STORE(k,v) atomicStore(k, v, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease)

void main() {
    string readBuf = malloc(4096);

    io r;
    int64_t startTime;

    if (ThreadId == 0) rerunProgram = RERUN_ON_IO;

    // This should do 512 accept+reads at a time and process them as they become ready.
    // Ditto for the writes and closes.
    // r = acceptReadBatch(listen_fd, conn_fds_i32a, reads_str_array);
    // conn_count = awaitIO(r).x;
    // pfor(i, conn_count, {
    //   process_req(i, conn_fds_i32a, reads_str_array, writes_str_array);
    // });
    // writeCloseBatch(conn_fds_buf, conn_count, writes_str_array);
    //

    stateMachine m = loadStateMachine(s_Init);
//    atomicAdd(programReturnValue, 1);
//    if (ThreadId == 0) {
//        while(programReturnValue < ThreadCount);
//        startTime = microTimeSync();
//        STORE(io_pad_8, 0);
//        STORE(programReturnValue, 0);
//    }
//    while (LOAD(programReturnValue) != 0);
//    uint32_t idx = 0;
//    while (LOAD(io_pad_8) == 0) {
//        idx++;
//        if (ThreadId == 0 && (idx % 4096 == 0)) {
//            int64_t now = microTimeSync();
//            if (now - startTime > 10000) {
//                STORE(io_pad_8, 1);
//            }
//            idx = 1;
//        }
//        barrier();
        switch (getState(m)) {
            case s_Init:
                if (ThreadId == 0) FREE_ALL(log("Starting server on port 8080..."));
                setAttr(m, a_Server, listenSync(8080));
                setAttr(m, a_HeapStart, heapPtr);
                setState(m, s_Accept);
                if (ThreadId == 0) FREE_ALL(log("Server running, accepting connections."));
//                break;

            case s_Accept:
                heapPtr = getI32Attr(m, a_HeapStart);
                fromIOPtr = fromIOStart;
                toIOPtr = toIOStart;
                setAttr(m, a_ConnectionIO, acceptAndRecv(getAttr(m, a_Server), readBuf));
                setState(m, s_WaitingConn);
//                break;

            case s_WaitingConn:
                r = getIOAttr(m, a_ConnectionIO);
                if (pollIO(r)) {
                    string req;
                    socket conn = awaitIO2(r, req);
                    string response = process(req);
                    setAttr(m, a_WriteIO, sendAndClose(conn, response));
                    setState(m, s_Closing);
                }
                break;

            case s_Closing:
                r = getIOAttr(m, a_WriteIO);
                if (pollIO(r)) {
                    awaitIO(r);
                    setState(m, s_Accept);
                }
                break;
        }
//    }
    saveStateMachine(m);
}


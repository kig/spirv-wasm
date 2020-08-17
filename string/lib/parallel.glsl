/*
    Sketching macros to help control program shape
    ===

    The idea is that you have four major types of program execution:
        1. All threads working on the same task.
        2. Threadgroups working on different concurrent tasks (e.g. producer-consumer queue.)
        3. Single thread in a threadgroup working on something sequential, rest of group idles.
        4. Single thread working on something sequential, rest of program idles.

    To help with autoscaling functions, I'm toying with the idea of an execution context
    that tells the runtime which mode of execution is in use.

    The goal with this would be to avoid correctness issues when trying to run parallel code in
    a single-thread context, and to avoid performance issues when running in multithreaded contexts.
    Without having to explicitly choose a function to match the context.

    E.g. memcpy: single-threaded memcpy has low performance, multi-threaded memcpy doesn't work if only one thread is running.
    If you don't know the execution context at runtime, you'd have to carefully build your program to use the correct type of
    memcpy at each point. If you know the context, you can use all available threads for it.

    // Run in the default context: group-wide execution. It'd be better to default to all-threads even though it requires convergent control flow.
    memcpy(dst, src, 400 * Mebi);

    // Parallel single-threaded memcpys with different params for different threads.
    single(memcpy(dst + ThreadLocalId * 4*kibi, src + ThreadLocalId * 8*kibi, 4*kibi));

    // Allow only ThreadId 0 execute the memcpy
    unique(memcpy(dst, src, 400 * Mebi));

    // Allow only first thread through the gate to execute the memcpy
    uniqueany(memcpy(dst, src, 400 * Mebi));

    // Do the copy across all program threads (requires that all threads arrive at this point in control flow)
    all(memcpy(dst, src, 400 * Mebi));

    // Use thread groups 4..15 to do the memcpy. Requires that thread groups 4..15 all arrive at this point.
    groups(4, 16, memcpy(dst, src, 400 * Mebi));


    Implement producer-consumer system:

    shared string readResult;
    shared string processingResult;

    groups(0, 4, {
        bool done = false;
        while (!done) {
            freeIO(
                fillLock(ReadBuffers[ActiveGroupId].status, {
                    io r = read(file, readOffset + ActiveGroupID*readSize, readSize, ReadBuffers[ActiveGroupId].buffer);
                    readOffset += ActiveGroupCount * readSize;
                    readResult = awaitIO(r);
                    ReadBuffers[ActiveGroupId].length = strLen(readResult);
                });
                done = (strLen(readResult) < readSize);
            )
        }
    })

    groups(4, 4, {
        bool done = false;
        while (!done) {
            freeAll(
                drainLock(ReadBuffers[ActiveGroupId].status, {
                    readResult = ReadBuffers[ActiveGroupId].buffer;
                    readResult.y = readResult.x + ReadBuffers[ActiveGroupId].length;
                    processingResult = processResult(readResult);
                });
                fillLock(WriteBuffers[ActiveGroupId].status, {
                    memcpy(WriteBuffers[ActiveGroupId].buffer, processingResult);
                    WriteBuffers[ActiveGroupId].length = strLen(processingResult);
                    WriteBuffers[ActiveGroupId].done = strLen(readResult) < readSize;
                });
                done = (strLen(readResult) < readSize);
            )
        }
    })

    groups(8, 4, {
        bool done = false;
        while (!done) {
            freeIO(
                drainLock(WriteBuffers[ActiveGroupId].status, {
                    writeSync(outFile, writeOffset + ActiveGroupID*writeSize, writeSize, string(0, WriteBuffers[ActiveGroupId].length) + WriteBuffers[ActiveGroupId].buffer.x);
                    writeOffset += ActiveGroupCount * writeSize;
                    done = WriteBuffers[ActiveGroupId].done;
                });
            )
        }
    })


const int32_t Empty = 0;
const int32_t Access = 1;
const int32_t Full = 2;

#define fillLock(var, f) while (atomicCompSwap(var, Empty, Access) != Empty); f; var = Full;
#define drainLock(var, f) while (atomicCompSwap(var, Full, Access) != Full); f; var = Empty;

*/

shared int32_t _ctx_;
shared int32_t _start_group_;
shared int32_t _end_group_;

#define GlobalBarrierLock io_pad_15
#define ActiveGroupCount (_end_group_ - _start_group_)
#define ActiveGroupId (ThreadGroupId - _start_group_)

#define Gibi 1073741824
#define Mebi 1048576
#define kibi 1024

#define Giga 1000000000
#define Mega 1000000
#define kilo 1000

#define CTX_GROUP_THREADS 0
#define CTX_SINGLE_THREAD 1
#define CTX_ALL_THREADS 2
#define CTX_MULTIGROUP_THREADS 3

#define widefor(t, i, start, end) for ( \
    t      i = t(_ctx_ == CTX_SINGLE_THREAD ? 0 : (_ctx_ == CTX_ALL_THREADS ? ThreadId    : ThreadLocalID   )) + (start), \
    t _incr_ = t(_ctx_ == CTX_SINGLE_THREAD ? 1 : (_ctx_ == CTX_ALL_THREADS ? ThreadCount : ThreadLocalCount));           \
    i < (end); i += _incr_)

#define allfor(t, i, start, end) for (t i = ThreadID + start; i < end; i += ThreadCount)
#define groupfor(t, i, start, end) for (t i = ThreadLocalID + start; i < end; i += ThreadLocalCount)

void copyFromIOToHeap(ptr_t src, ptr_t dst, size_t len) {
    widefor(ptr_t, i,         0, len/16) i64v2heap[dst/16+i] = i64v2fromIO[src/16+i];
    widefor(ptr_t, i, len/16*16, len   ) heap[dst+i]         = fromIO[src+i];
}

void globalBarrier() {
    atomicAdd(GlobalBarrierLock, 1);
    if (ThreadId == 0) {
        while (GlobalBarrierLock < ThreadCount);
        GlobalBarrierLock = 0;
    }
    while (GlobalBarrierLock != 0);
}

void deviceBarrier() {
    controlBarrier(gl_ScopeDevice, gl_ScopeDevice, gl_StorageSemanticsBuffer | gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
}

#define unique(f) { \
    atomicMax(CtxLock, 1); \
    if (ThreadId == 0) { \
        int32_t _prevctx_ = _ctx_; \
        _ctx_ = CTX_SINGLE_THREAD; \
        { f; } \
        CtxLock = 0;
        _ctx_ = _prevctx_; \
    } \
    while(CtxLock != 0); \
}

#define uniqueany(f) { \
    if (atomicMax(CtxLock, 1) == 0) { \
        int32_t _prevctx_ = _ctx_; \
        _ctx_ = CTX_SINGLE_THREAD; \
        { f; } \
        CtxLock = 0;
        _ctx_ = _prevctx_; \
    } \
    while(CtxLock != 0); \
}

#define single(f) { \
    int32_t _prevctx_ = _ctx_; \
    _ctx_ = CTX_SINGLE_THREAD; \
    { f; } \
    _ctx_ = _prevctx_; \
}

#define all(f) { \
    int32_t _prevctx_ = _ctx_; \
    _ctx_ = CTX_ALL_THREADS; \
    { f; } \
    _ctx_ = _prevctx_; \
}

#define groupunique(f) { \
    if (ThreadLocalId == 0) { \
        int32_t _prevctx_ = _ctx_; \
        _ctx_ = CTX_SINGLE_THREAD; \
        { f; } \
        _ctx_ = _prevctx_; \
    } \
    barrier(); \
}

#define groups(start, count, f) { \
    if (ThreadGroupId >= start && ThreadGroupId < start + end) { \
        int32_t _prevctx_ = _ctx_; \
        int32_t _prevstart_ = _start_group_; \
        int32_t _prevend_ = _end_group_; \
        _ctx_ = CTX_MULTIGROUP_THREADS; \
        _start_group_ = start;
        _end_group_ = start + end;
        { f; } \
        _ctx_ = _prevctx_; \
        _start_group_ = _prevstart_; \
        _end_group_ = _prevend_; \
    } \
    barrier(); \
}

#define group(n) if (ThreadGroupId == n)
#define thread(n) if (ThreadId == n)
#define threadlocal(n) if (ThreadLocalId == n)

#define widereduce(sum, i, arr, f) widefor (size_t, _i_, 0, arrLen(arr)) { \
    i = aGet(arr, _i_); \
    f; \
}


//int x = 0, i = 0;
//widereduce(x, i, myArray, atomicAdd(x, i));


#define treeReduce(array, sum, a, b, reduce) for (int _width_ = 1; _width_ < arrLen(array); _width_ *= 2) { \
    for (int _i_ = ThreadId; _i_ < arrLen(array); _i_ += ThreadCount) { \
        if (_i_ & (_width_ * 2 - 1) == 0) { \
            alloc_t a = aGet(array, _i_); \
            alloc_t b = aGet(array, _i_ + _width_); \
            reduce; \
            aSet(array, _i_, sum); \
        } \
    } \
    controlBarrier(gl_ScopeDevice, gl_ScopeDevice, gl_StorageSemanticsBuffer | gl_StorageSemanticsShared, gl_SemanticsAcquireRelease); \
}

#define groupTreeReduce(array, sum, a, b, reduce) for (int _width_ = 1; _width_ < arrLen(array); _width_ *= 2) { \
    for (int _i_ = ThreadLocalId; _i_ < arrLen(array); _i_ += ThreadLocalCount) { \
        if (_i_ & (_width_ * 2 - 1) == 0) { \
            alloc_t a = aGet(array, _i_); \
            alloc_t b = aGet(array, _i_ + _width_); \
            reduce; \
            aSet(array, _i_, sum); \
        } \
    } \
    barrier(); \
}


//treeReduce(array, a, a, b, i32heap[a.x] = i32heap[a.x] + i32heap[b.x]);
//
//alloc_t reduce(alloc_t a, alloc_t b) {
//    i32heap[a.x] = i32heap[a.x] + i32heap[b.x];
//    return a;
//}
//
//for (int width = 1; width < arrLen(array); width *= 2) {
//    for (int i = ThreadId; i < arrLen(array); i += ThreadCount) {
//        if (i & (width * 2 - 1) == 0) {
//            aSet(array, i, reduce(aGet(array, i), aGet(array, i + width));
//        }
//    }
//}

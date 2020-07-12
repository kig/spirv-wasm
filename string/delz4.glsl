#!/usr/bin/env gls

ThreadLocalCount = 8;
ThreadGroupCount = 1;

HeapSize = 250000;
ToIOSize = 250000;
FromIOSize = 250000;

#define LZ4_GROUP_SIZE 8

#include "file.glsl"
#include "lz4.glsl"

shared string compressed;

void main() {

    string filename = aGet(argv, 1);

    if (ThreadId == 0) {
        heapPtr = 0;
        fromIOPtr = 0;
        toIOPtr = 0;
        FREE_IO(
            io r = read(filename, malloc(1000000));
            compressed = awaitIO(r, true);
        )
    }
    barrier();
    ptr_t end = lz4DecompressFramesFromIOToHeap(compressed, string(0, TotalHeapSize));
    barrier();
    if (ThreadId == 0) {
        heapPtr = 0;
        fromIOPtr = 0;
        toIOPtr = 0;
        print(string(0, end));
    }

}


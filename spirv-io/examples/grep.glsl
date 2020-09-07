#!/usr/bin/env gls

ThreadLocalCount = 256;
ThreadGroupCount = 64;

#define LZ4_GROUP_SIZE 8
#define LZ4_STREAM_BLOCK_SIZE 8192

#include <file.glsl>
#include <lz4.glsl>

shared int done;
shared int64_t wgOff;
shared string wgBuf;
shared int32_t decompressedSize;
shared bool isCompressed;

void addHit(int32_t k, int32_t off, inout bool found) {
    i32fromIO[atomicAdd(groupHeapPtr, 4)/4] = int32_t(k) + off;
    found = true;
}

bool grepBuffer(int32_t blockSize, string buf, string pattern, char p, int32_t off) {
    bool found = false;
    for (size_t i = 0, l = strLen(buf); i < blockSize; i+=32) {
        ptr_t idx = buf.x + i;
        i64vec4 v = i64v4heap[idx / 32];
        for (size_t j = 0, k = i, jdx = idx; j < 64; j += 8, idx++, k++, jdx++) {
            i8vec4 u = i8vec4((v >> j) & 0xff);
            if (any(equal(u, i8vec4(p)))) {
                if (k < l && p == u.x && startsWith(string(jdx, buf.y), pattern)) addHit(k, off, found);
                if (k+8 < l && p == u.y && startsWith(string(jdx+8, buf.y), pattern)) addHit(k + 8, off, found);
                if (k+16 < l && p == u.z && startsWith(string(jdx+16, buf.y), pattern)) addHit(k + 16, off, found);
                if (k+24 < l && p == u.w && startsWith(string(jdx+24, buf.y), pattern)) addHit(k + 24, off, found);
            }
        }
    }
    return found;
}


void main() {

    if (arrLen(argv) < 3) {
        if (ThreadId == 0) eprintln("USAGE: grep.glsl pattern file");
        return;
    }

    string pattern = aGet(argv, 1);
    string filename = aGet(argv, 2);

    if (ThreadId == 0) {
        Stat st = statSync(filename);
        programReturnValue = (st.error == 0) ? 1 : 2;
        // readaheadSync(filename, 0, st.st_size);
    }
    while (programReturnValue == 0); // Wait for first thread.

    if (programReturnValue == 2) {
        if (ThreadId == 0) eprintln(concat("File not found: ", filename));
        return;
    }

    int32_t patternLength = strLen(pattern);
    int32_t blockSize = HeapSize - (((patternLength+31) / 32) * 32);
    int32_t wgBufSize = ThreadLocalCount * blockSize + patternLength;

    if (ThreadLocalId == 0) {
        done = 0;
        wgOff = int64_t(ThreadGroupId * ThreadLocalCount) * int64_t(blockSize);
        isCompressed = true;
    }

    bool found = false;
    char p = heap[pattern.x];

    ptr_t hitStart = 0;

    while (done == 0) {
        FREE(FREE_IO(
            barrier(); memoryBarrier();

            if (ThreadLocalId == 0) {
                fromIOPtr = groupHeapStart;
                toIOPtr = groupHeapStart;

                io r = read(filename, wgOff, wgBufSize, string(groupHeapStart, groupHeapStart + (HeapSize * ThreadLocalCount)), IO_COMPRESS_LZ4_BLOCK_STREAM | LZ4_STREAM_BLOCK_SIZE);
                wgBuf = awaitIO(r, true, decompressedSize, isCompressed);

                if (decompressedSize != wgBufSize) {
                    done = (decompressedSize == 0) ? 2 : 1;
                }
                groupHeapPtr = groupHeapStart;
                hitStart = groupHeapPtr;
            }

            barrier(); memoryBarrier();

            if (done == 2) break;

            if (isCompressed) {
                for (int32_t i = 0; i < 128; i += ThreadLocalCount/LZ4_GROUP_SIZE) {
                    lz4DecompressBlockStreamFromIOToHeap(i + ThreadLocalId/LZ4_GROUP_SIZE, LZ4_STREAM_BLOCK_SIZE, wgBuf, string(groupHeapStart, groupHeapStart + decompressedSize));
                }
            } else {
                copyFromIOToHeap(
                    string(groupHeapStart + ThreadLocalId * HeapSize, groupHeapStart + (ThreadLocalId+1) * HeapSize),
                    string(groupHeapStart + ThreadLocalId * HeapSize, groupHeapStart + (ThreadLocalId+1) * HeapSize)
                );
            }

            if (ThreadLocalId == 0) {
                wgBuf = string(groupHeapStart, groupHeapStart + decompressedSize);
            }

            barrier(); memoryBarrier();

            string buf = string(
                min(wgBuf.y, wgBuf.x + ThreadLocalId * blockSize),
                min(wgBuf.y, wgBuf.x + (ThreadLocalId+1) * blockSize + patternLength)
            );

            bool blockFound = grepBuffer(blockSize, buf, pattern, p, ThreadLocalId * blockSize);
            found = found || blockFound;

            barrier(); memoryBarrier();

            if (ThreadLocalId == 0) {
                fromIOPtr = groupHeapStart;
                toIOPtr = groupHeapStart;
                ptr_t start = hitStart / 4;
                ptr_t end = groupHeapPtr / 4;

                if (start != end) {
                    heapPtr = groupHeapStart;
                    for (int j = start; j < end; j++) {
                        str(int64_t(i32fromIO[j]) + wgOff);
                        _w('\n');
                    }
                    print(string(groupHeapStart, heapPtr));
                }

                wgOff += int64_t(ThreadCount * blockSize);
            }

            barrier(); memoryBarrier();
        ))
    }

    atomicMin(programReturnValue, found ? 0 : 1);
}


#!/usr/bin/env gls

ThreadLocalCount = 256;
ThreadGroupCount = 64;

#define LZ4_GROUP_SIZE 8
#define LZ4_STREAM_BLOCK_SIZE 16384

#include "file.glsl"
#include "lz4.glsl"

shared int done;
shared int64_t wgOff;
shared string wgBuf;
shared int32_t decompressedSize;
shared ptr_t groupHeapPtr;
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

    string pattern = aGet(argv, 1);
    string filename = aGet(argv, 2);

    if (ThreadId == 0) programReturnValue = 1;
    controlBarrier(gl_ScopeDevice, gl_ScopeDevice, 0, 0);

    int32_t patternLength = strLen(pattern);
    int32_t blockSize = HeapSize-(((patternLength+31) / 32) * 32);
    ptr_t tgHeapStart = HeapSize * ThreadLocalCount * ThreadGroupId;
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
                fromIOPtr = tgHeapStart;
                toIOPtr = tgHeapStart;

                io r = read(filename, wgOff, wgBufSize, string(tgHeapStart, tgHeapStart + (HeapSize * ThreadLocalCount))); //, IO_COMPRESS_LZ4_BLOCK_STREAM | LZ4_STREAM_BLOCK_SIZE | IO_COMPRESS_SPEED_7);
                //io r = read(filename, wgOff, wgBufSize, string(tgHeapStart, tgHeapStart + (HeapSize * ThreadLocalCount)), IO_COMPRESS_LZ4);
                wgBuf = awaitIO(r, true, decompressedSize, isCompressed);

                if (decompressedSize != wgBufSize) {
                    done = (decompressedSize == 0) ? 2 : 1;
                }
                groupHeapPtr = tgHeapStart;
                hitStart = groupHeapPtr;
            }

            barrier(); memoryBarrier();

            if (done == 2) break;

            if (isCompressed) {
                for (int32_t i = 0; i < 128; i += ThreadLocalCount/LZ4_GROUP_SIZE) {
                    lz4DecompressBlockStreamFromIOToHeap(i + ThreadLocalId/LZ4_GROUP_SIZE, LZ4_STREAM_BLOCK_SIZE, wgBuf, string(tgHeapStart, tgHeapStart + decompressedSize));
                }
            } else {
                copyFromIOToHeap(
                    string(tgHeapStart + ThreadLocalId * HeapSize, tgHeapStart + (ThreadLocalId+1) * HeapSize),
                    string(tgHeapStart + ThreadLocalId * HeapSize, tgHeapStart + (ThreadLocalId+1) * HeapSize)
                );
            }
            
            if (ThreadLocalId == 0) {
                wgBuf = string(tgHeapStart, tgHeapStart + decompressedSize);
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
                fromIOPtr = tgHeapStart;
                toIOPtr = tgHeapStart;
                ptr_t start = hitStart / 4;
                ptr_t end = groupHeapPtr / 4;
                
                if (start != end) {
                    heapPtr = tgHeapStart;
                    for (int j = start; j < end; j++) {
                        str(int64_t(i32fromIO[j]) + wgOff);
                        _w('\n');
                    }
                    print(string(tgHeapStart, heapPtr));
                }

                wgOff += int64_t(ThreadCount * blockSize);
                
            }

            barrier(); memoryBarrier();
        ))
    }

    atomicMin(programReturnValue, found ? 0 : 1);
}


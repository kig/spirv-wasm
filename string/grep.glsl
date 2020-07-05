layout ( local_size_x = 256, local_size_y = 1, local_size_z = 1 ) in;

#include "file.glsl"

shared int done;
shared int64_t wgOff;
shared string wgBuf;
shared ptr_t groupHeapPtr;

bool startsWithIO(string s, string pattern) {
    if (strLen(pattern) > strLen(s)) return false;
    for (ptr_t i = s.x, j = pattern.x; i < s.y && j < pattern.y; i++, j++) {
        if (fromCPU[i] != heap[j]) return false;
    }
    return true;
}

void addHit(int32_t k, int32_t off, inout bool found) {
    i32heap[atomicAdd(groupHeapPtr, 4)/4] = int32_t(k) + off;
    found = true;
}

bool grepBuffer(int32_t blockSize, string buf, string pattern, char p, int32_t off) {
    bool found = false;
    /*
    for (size_t i = 0, l = strLen(buf); i < blockSize; i++) {
        ptr_t idx = buf.x + i;
        if (i < l && p == fromCPU[idx] && startsWithIO(string(idx, buf.y), pattern)) addHit(i, off, found);
    }
    return found;
    */
    for (size_t i = 0, l = strLen(buf); i < blockSize; i+=32) {
        ptr_t idx = buf.x + i;
        i64vec4 v = i64v4fromCPU[idx / 32];
        for (size_t j = 0, k = i, jdx = idx; j < 64; j += 8, idx++, k++, jdx++) {
            i8vec4 u = i8vec4((v >> j) & 0xff);
            if (any(equal(u, i8vec4(p)))) {
                if (k < l && p == u.x && startsWithIO(string(jdx, buf.y), pattern)) addHit(k, off, found);
                if (k+8 < l && p == u.y && startsWithIO(string(jdx+8, buf.y), pattern)) addHit(k + 8, off, found);
                if (k+16 < l && p == u.z && startsWithIO(string(jdx+16, buf.y), pattern)) addHit(k + 16, off, found);
                if (k+24 < l && p == u.w && startsWithIO(string(jdx+24, buf.y), pattern)) addHit(k + 24, off, found);
            }
        }
    }
    return found;
}

void main() {
    initGlobals();

    string pattern = aGet(argv, 1);
    string filename = aGet(argv, 2);

    if (ThreadID == 0) programReturnValue = 1;
    controlBarrier(gl_ScopeDevice, gl_ScopeDevice, 0, 0);

    int32_t patternLength = strLen(pattern);
    int32_t blockSize = HEAP_SIZE-(((patternLength+31) / 32) * 32);

    ptr_t tgHeapStart = HEAP_SIZE * ThreadLocalCount * ThreadGroupID;

    int32_t wgBufSize = ThreadLocalCount * blockSize + patternLength;

    if (ThreadLocalID == 0) {
        done = 0;
        wgOff = int64_t(ThreadGroupID * ThreadLocalCount) * int64_t(blockSize);
    }

    bool found = false;
    char p = heap[pattern.x];

    while (done == 0) {
        FREE(FREE_IO(
            barrier(); memoryBarrier();

            if (ThreadLocalID == 0) {
                fromCPUPtr = tgHeapStart;
                
                io r = read(filename, wgOff, wgBufSize, string(tgHeapStart, tgHeapStart + wgBufSize));
                wgBuf = awaitIO(r, true);
                
                if (strLen(wgBuf) != wgBufSize) {
                    done = strLen(wgBuf) == 0 ? 2 : 1;
                }
                groupHeapPtr = tgHeapStart;
            }

            barrier(); memoryBarrier();

            if (done == 2) break;
            
            string buf = string(
                min(wgBuf.y, wgBuf.x + ThreadLocalID * blockSize),
                min(wgBuf.y, wgBuf.x + (ThreadLocalID+1) * blockSize + patternLength)
            );

            bool blockFound = grepBuffer(blockSize, buf, pattern, p, ThreadLocalID * blockSize);
            found = found || blockFound;

            barrier(); memoryBarrier();

            if (ThreadLocalID == 0) {
                fromCPUPtr = tgHeapStart;
                toCPUPtr = tgHeapStart;
                ptr_t start = tgHeapStart / 4;
                ptr_t end = groupHeapPtr / 4;
                
                if (start != end) {
                    heapPtr = groupHeapPtr;
                    for (int j = start; j < end; j++) {
                        str(int64_t(i32heap[j]) + wgOff);
                        _w('\n');
                    }
                    print(string(groupHeapPtr, heapPtr));
                }

                wgOff += int64_t(ThreadCount * blockSize);
                
            }

            barrier(); memoryBarrier();
        ))
    }

    atomicMin(programReturnValue, found ? 0 : 1);
}


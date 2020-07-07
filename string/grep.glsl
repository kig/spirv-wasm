layout ( local_size_x = 224, local_size_y = 1, local_size_z = 1 ) in;
#include "file.glsl"

shared int done;
shared int64_t wgOff;
shared string wgBuf;
shared int32_t decompressedSize;
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


#define LZ4_GROUP_SIZE 4
#define LZ4_STREAM_BLOCK_SIZE 8192

void lz4DecompressFromCPUToHeap(int32_t blockIndex, int32_t blockSize, string cmp, string dst) {
    if (blockIndex >= 128) return;

    ptr_t i = cmp.x + 128 * 4;
    int32_t len = 0;
    for (int32_t b=0; b<=blockIndex && b<128; b++) {
        len = i32fromCPU[cmp.x/4 + b];
        if (b == blockIndex || len <= 0) break;
        i += (len+7)/8*8;
    }
    if (len <= 0 || i >= cmp.y) {
        return;
    }
    ptr_t outputStart = dst.x + blockSize * blockIndex;
    int32_t subId = (ThreadLocalID % LZ4_GROUP_SIZE);
    
    for (ptr_t be=min(cmp.y, i+len), j=outputStart, dbe=min(dst.y, outputStart+blockSize); i<be && j<dbe;) {
        uint8_t token = u8fromCPU[i++];
        int32_t litLen = (int32_t(token) >> 4) & 0xf;
        uint8_t c = uint8_t(litLen | (litLen << 4));
        int32_t matchLen = int32_t(token) & 0xf;
        while (c == 0xff) {
            c = u8fromCPU[i++];
            litLen += int32_t(c);
        }
        {
            int32_t k = 0;
            for (; k < litLen-(LZ4_GROUP_SIZE-1); k+=LZ4_GROUP_SIZE) u8heap[k+j+subId] = u8fromCPU[k+i+subId];
            if (k < litLen && subId < litLen-k) u8heap[k+j+subId] = u8fromCPU[k+i+subId];
            i += litLen;
            j += litLen;
        }

        if (((j-outputStart) & 8191) == 0 && matchLen == 0) { // End of LZ4 block
            continue;
        }

        int32_t matchOff = (int32_t(u8fromCPU[i  ]) << 0)
                         | (int32_t(u8fromCPU[i+1]) << 8);
        i += 2;
        c = uint8_t(matchLen | (matchLen << 4));
        matchLen += 4;
        while (c == 0xff) {
            c = u8fromCPU[i++];
            matchLen += int32_t(c);
        }
        ptr_t m = j - matchOff;
        {
            int32_t k = 0;
            int32_t maxSubSize = min(LZ4_GROUP_SIZE, matchOff);
            if (subId < maxSubSize) {
                for (; k < matchLen-(maxSubSize-1); k+=maxSubSize) u8heap[k+j+subId] = u8heap[k+m+subId];
                if (k < matchLen && subId < matchLen-k) u8heap[k+j+subId] = u8heap[k+m+subId];
            }
            j += matchLen;
        }
    }
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

    ptr_t hitStart = (tgHeapStart + decompressedSize + 3) / 4 * 4;

    while (done == 0) {
        FREE(FREE_IO(
            barrier(); memoryBarrier();

            if (ThreadLocalID == 0) {
                fromCPUPtr = tgHeapStart;

                io r = read(filename, wgOff, wgBufSize, string(tgHeapStart, tgHeapStart + (HEAP_SIZE * ThreadLocalCount)));
                wgBuf = awaitIO(r, true, decompressedSize);

                if (decompressedSize != wgBufSize) {
                    done = decompressedSize == 0 ? 2 : 1;
                }
                groupHeapPtr = (tgHeapStart + decompressedSize + 3) / 4 * 4;
                hitStart = groupHeapPtr;
            }

            barrier(); memoryBarrier();

            if (done == 2) break;

            for (int32_t i = 0; i < 128; i += ThreadLocalCount/LZ4_GROUP_SIZE) {
                lz4DecompressFromCPUToHeap(i + ThreadLocalID/LZ4_GROUP_SIZE, LZ4_STREAM_BLOCK_SIZE, wgBuf, string(tgHeapStart, tgHeapStart + decompressedSize));
            }

            if (ThreadLocalID == 0) {
                wgBuf = string(tgHeapStart, tgHeapStart + decompressedSize);
            }
            
            barrier(); memoryBarrier();

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
                ptr_t start = hitStart / 4;
                ptr_t end = groupHeapPtr / 4;
                
                if (start != end) {
                    heapPtr = tgHeapStart;
                    for (int j = start; j < end; j++) {
                        str(int64_t(i32heap[j]) + wgOff);
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


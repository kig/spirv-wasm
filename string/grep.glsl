layout ( local_size_x = 255, local_size_y = 1, local_size_z = 1 ) in;

#define HEAP_SIZE 4096

#include "file.glsl"

shared int done;
shared int wgOff;
shared string wgBuf;

bool grepBuffer(int blockSize, string buf, string pattern, char p, int wgOff) {
    bool found = false;
    for (size_t i = 0, l = strLen(buf); i < blockSize; i+=32) {
        ptr_t idx = buf.x + i;
        i64vec4 v = i64v4heap[idx / 32];
        for (size_t j = 0, k = i, jdx = idx; j < 64; j += 8, idx++, k++, jdx++) {
            i8vec4 u = i8vec4((v >> j) & 0xff);
            if (any(equal(u, i8vec4(p)))) {
                if (k < l && p == u.x && startsWith(string(jdx, buf.y), pattern)) { i32heap[heapPtr/4] = int32_t(k); heapPtr+=4; found = true;  }
                if (k+8 < l && p == u.y && startsWith(string(jdx+8, buf.y), pattern)) { i32heap[heapPtr/4] = int32_t(k + 8); heapPtr+=4; found = true;  }
                if (k+16 < l && p == u.z && startsWith(string(jdx+16, buf.y), pattern)) { i32heap[heapPtr/4] = int32_t(k + 16); heapPtr+=4; found = true;  }
                if (k+24 < l && p == u.w && startsWith(string(jdx+24, buf.y), pattern)) { i32heap[heapPtr/4] = int32_t(k + 24); heapPtr+=4; found = true;  }
            }
        }
    }
    return found;
}

void main() {
    initGlobals();

    string pattern = aGet(argv, 1);
    string filename = aGet(argv, 2);

    if (ThreadID == 0) {
        FREE(
            println(concat("Searching for pattern ", pattern));
            println(concat("In file ", filename));
        )
        /*
            println(concat("Thread count ", str(ThreadCount)));
            println(concat("Thread groups ", str(ThreadGroupCount)));
            println(concat("Local thread count ", str(ThreadLocalCount)));
            println(concat("Total heap ", str(ThreadCount * HEAP_SIZE)));
            println(concat("IO size ", str(ThreadLocalCount * HEAP_SIZE)));
        )
        */
        atomicAdd(programReturnValue, 1);
    }
    while(programReturnValue == 0);

    int patternLength = strLen(pattern);
    int blockSize = HEAP_SIZE-(((patternLength+31) / 32) * 32);

    int tgHeapStart = HEAP_SIZE * ThreadLocalCount * ThreadGroupID;
    int tgHeapStart2 = HEAP_SIZE * ThreadLocalCount * ThreadGroupID + ThreadLocalCount * HEAP_SIZE/2;

    int wgBufSize = ThreadLocalCount * blockSize + patternLength;
    
    if (ThreadLocalID == 0) {
        done = 0;
        wgOff = ThreadGroupID * ThreadLocalCount * blockSize;
    }

    bool found = false;
    int startp = toIndexPtr(heapPtr);

    char p = heap[pattern.x];

/*
    io r0, r1;
    if (ThreadLocalID == 0) {
        wgBuf = string(tgHeapStart, tgHeapStart + wgBufSize);
        r0 = read(filename, wgOff, wgBufSize, wgBuf);
    }
*/

    while (done == 0) {
        FREE(FREE_IO(
            barrier();
            memoryBarrier();

            if (ThreadLocalID == 0) {
                /*
                r1 = read(filename, wgOff + ThreadCount * blockSize, wgBufSize, 
                    wgBuf.x == tgHeapStart
                    ? string(tgHeapStart2, tgHeapStart2 + wgBufSize)
                    : string(tgHeapStart, tgHeapStart + wgBufSize)
                );
                wgBuf = awaitIO(r0);
                r0 = r1;
                */
                wgBuf = readSync(filename, wgOff, wgBufSize, string(tgHeapStart, tgHeapStart + wgBufSize));
                
                //if (wgOff > 1000000000) wgBuf.y = wgBuf.x;

                if (strLen(wgBuf) != wgBufSize) {
                    done = strLen(wgBuf) == 0 ? 2 : 1;
                }
            }

            barrier();
            memoryBarrier();

            if (done == 2) break;
            
            string buf = string(
                min(wgBuf.y, wgBuf.x + ThreadLocalID * blockSize),
                min(wgBuf.y, wgBuf.x + (ThreadLocalID+1) * blockSize + patternLength)
            );

            int start = startp;
            heapPtr = startp * 4;

            found = grepBuffer(blockSize, buf, pattern, p, wgOff) || found;

            int end = heapPtr / 4;

            barrier();
            memoryBarrier();

            for (int j = start; j < end; j++) {
                str(uint(i32heap[j] + ThreadLocalID * blockSize + wgOff));
                _w('\n');
            }

            barrier();
            memoryBarrier();

            if (end * 4 != heapPtr) print(string(end*4, heapPtr));

            barrier();
            memoryBarrier();

            if (ThreadLocalID == 0) {
                // atomicMax(io_pad_3, wgOff + strLen(wgBuf));
                wgOff += ThreadCount * blockSize;
            }
        ))
    }

    atomicMin(programReturnValue, found ? 0 : 1);
}


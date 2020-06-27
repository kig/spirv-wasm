layout ( local_size_x = 16, local_size_y = 1, local_size_z = 1 ) in;

#define HEAP_SIZE 8192

#include "file.glsl"

shared int done;
shared int wgOff;
shared string wgBuf;

void main() {
    initGlobals();

    string filename = aGet(argv, 2);
    string pattern = aGet(argv, 1);

    if (ThreadLocalID == 0) done = 0;

    if (ThreadID == 0) {
        FREE(
            println(concat("Searching for pattern ", pattern));
            println(concat("In file ", filename));
        )
        setReturnValue(1);
    }
    while(ioRequests[0].status == 0);

    int blockSize = 8000;
    int plen = strLen(pattern);

    int off = int(ThreadLocalID) * blockSize;

    int wgBufSize = ThreadLocalCount * blockSize + plen;
    
    wgOff = ThreadGroupID * ThreadLocalCount * blockSize;
    wgBuf = string(HEAP_SIZE * ThreadLocalCount * ThreadGroupID, 0);

    //if (ThreadLocalID == 0) println(concat("Block size ", str(wgBufSize), ", group offset ", str(wgOff)));

    int starth = heapPtr;

    bool found = false;
    int startp = i32heapPtr;

    /*
    io r0, r1;
    string wgBuf2 = wgBuf + string(wgBufSize, wgBufSize);
    if (ThreadLocalID == 0) {
        wgBuf.y = wgBuf.x + wgBufSize;
        r0 = read(filename, wgOff, wgBufSize, wgBuf);
    }
    */
    while (done == 0) {
        FREE(FREE_IO(
            barrier();
            memoryBarrier();

            if (ThreadLocalID == 0) {
                //wgBuf2.y = wgBuf2.x + wgBufSize;
                //r1 = read(filename, wgOff + wgBufSize, wgBufSize, wgBuf2);
                //wgBuf = awaitIO(r0);
                wgBuf.y = wgBuf.x + wgBufSize;
                wgBuf = readSync(filename, wgOff, wgBufSize, wgBuf);
                if (strLen(wgBuf) != wgBufSize) atomicAdd(done, strLen(wgBuf) == 0 ? 2 : 1);
                //r0 = r1;
                //wgBuf2 = wgBuf;
            }

            barrier();
            memoryBarrier();

            if (done == 2) break;
            
            string buf = string(
                min(wgBuf.y, wgBuf.x + ThreadLocalID * blockSize),
                min(wgBuf.y, wgBuf.x + (ThreadLocalID+1) * blockSize + plen)
            );

            int sx = buf.x;

            int start = startp;
            i32heapPtr = startp;
            for (int i = 0; i < blockSize; i++) {
                int idx = buf.x + i;
                if (strCmp(pattern, string(idx, buf.y)) == 0) {
                    i32heap[i32heapPtr++] = int32_t(wgOff + off + idx);
                    found = true;
                }
            }
            int end = i32heapPtr;

            barrier();

            heapPtr = starth;

            //FREE(println(str(ivec2(wgOff + off, blockSize + plen))));
            
            for (int j = start; j < end; j++) {
                str(uint(i32heap[j]));
                _w('\n');
            }

            if (starth != heapPtr) print(string(starth, heapPtr));
            
            if (ThreadLocalID == 0) {
                atomicMax(ioRequests[0].offset, wgOff + strLen(wgBuf));
                wgOff += ThreadCount * blockSize;
            }
        ))
    }

    atomicMin(ioRequests[0].status, found ? 0 : 1);
}


#!/usr/bin/env gls

ThreadLocalCount = 32;
ThreadGroupCount = 1;

HeapSize = 65536;
FromIOSize = 65536;
ToIOSize = 65536;

#include <file.glsl>

shared int eof;

void main() {

    int64_t blockSize = int64_t(HeapSize);

    int argc = arrLen(argv);
    for (int i = 1; i < argc; i++) {
        eof = 0;
        string filename = aGet(argv, i);
        int64_t block = 0;
        barrier();
        while (eof == 0) {
            FREE(
                string res;
                FREE_IO(
                    int64_t off = (block * int64_t(ThreadCount) + int64_t(ThreadId)) * blockSize;

                    io r = read(filename, off, size_t(blockSize), malloc(size_t(blockSize)));
                    barrier();

                    res = awaitIO(r);
                    barrier();
                )
                FREE_IO(
                    for (int i = 0; i < ThreadCount; i++) {
                        barrier();
                        if (i == ThreadId && strLen(res) > 0) {
                            print(res);
                        }
                    }

                    if (strLen(res) < size_t(blockSize)) atomicAdd(eof, 1);

                    block++;
                    barrier();
                )
            )
        }
    }

}


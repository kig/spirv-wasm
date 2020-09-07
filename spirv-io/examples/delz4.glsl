#!/usr/bin/env gls

ThreadLocalCount = 32;
ThreadGroupCount = 216;

TotalHeapSize =   2040000000;
TotalToIOSize =   1283886080;
TotalFromIOSize = 83886080;

// #define DEBUG

#ifdef DEBUG
#define DLOG(s) FREE_ALL(log(s))
#else
#define DLOG(s) {}
#endif

#include <file.glsl>

#define LZ4_GROUP_SIZE 32
ptr_t LZ4Literals = (TotalHeapSize - (218 * 4096) - (ThreadGroupId+1) * 2*4*4*ThreadLocalCount) / 16;
ptr_t LZ4Matches = (LZ4Literals + ThreadLocalCount);

#include <lz4.glsl>

#define BLOCK_COUNT 32
#define LZ4_BLOCK_COUNT 216

const size_t bsz = (1<<20);


const ptr_t compressedBlocksCount = (2*(LZ4_BLOCK_COUNT+1) * (1<<22)) / 4;
const stringArray compressedBlocks = stringArray((compressedBlocksCount + 1), (compressedBlocksCount + 1) + 18000);

const ptr_t uncompressedLengths = compressedBlocks.y;

const string readBuffer = string(0, (LZ4_BLOCK_COUNT+1) * (1<<22));

#define IsLastBlock runCount
#define WriteLength io_pad_6
#define ReadHeapOffset io_pad_7
#define StopReading io_pad_8
#define ReadCount io_pad_9
#define ReadBarrier io_pad_10
#define ReadOffset io_pad_12


void main() {

    string filename = aGet(argv, 1);

    int64_t parseOffset = 0;
    uint32_t blockLength = 0;

    LZ4FrameHeader header;

    int error;

    uint32_t totalLen = 0;
    int64_t totalUncompressedLen = 0;

    if (ThreadId == 0) {
        ReadCount = 0;
        ReadBarrier = 0;
        ReadOffset = 0;
        programReturnValue = 0;
    }
    while(programReturnValue != 0);
    bool firstBlock = true;

    barrier();

    uint32_t contentChecksum = 0;
    io writeIO;

    error = LZ4_OK;

    while (ReadCount == 0) {
        barrier(); memoryBarrier();
        StopReading = 0;
        ReadHeapOffset = 0;
        int32_t blockCount = 0;
        while (StopReading == 0) {
            if (ThreadLocalId == 0 && ThreadGroupId < BLOCK_COUNT) {
                heapPtr = TotalHeapSize - 4096*(ThreadGroupId+1);
                fromIOPtr = ThreadGroupId*bsz;
                toIOPtr = TotalToIOSize - 4096*(ThreadGroupId+1);

                io r;
                FREE_ALL(r = read(filename, ReadOffset + int64_t(ThreadGroupId*bsz), bsz, string(ThreadGroupId*bsz, (ThreadGroupId+1)*bsz)));
                string compressed = awaitIO(r, true);
                atomicAdd(ReadCount, strLen(compressed));
            }
            if (atomicAdd(programReturnValue, 1) == 0) { while (programReturnValue < ThreadCount); programReturnValue = 0; } while (programReturnValue != 0);
            for (int i = ThreadId + ReadHeapOffset/16, j = ThreadId; j < (ReadCount/16)+1; j+=ThreadCount, i+=ThreadCount) {
                i64v2heap[i] = i64v2fromIO[j];
            }
            memoryBarrier();barrier();
            atomicAdd(programReturnValue, 1);
            if (ThreadId == 0) {
                while (programReturnValue < ThreadCount);

                ReadHeapOffset += ReadCount;
                ReadOffset += ReadCount;

                DLOG(concat("ReadOffset ", str(ReadOffset)));
                DLOG(concat("reads done ", str(ReadCount), " / ", str(bsz * BLOCK_COUNT)));

                if (!firstBlock) awaitIO(writeIO);
                if (firstBlock) {
                    parseOffset = readLZ4FrameHeaderFromHeap(0, header, error);
                    DLOG(concat("parse offset ", str(parseOffset)));
                    firstBlock = false;
                    blockLength = 1;
                }
                DLOG(concat("ReadHeapOffset ", str(ReadHeapOffset)));
                while (int32_t(parseOffset) < ReadHeapOffset && blockLength > 0) {
                    blockLength = readU32heap(int32_t(parseOffset));
                    bool isCompressed = !getBit(blockLength, 31);
                    blockLength = unsetBit(blockLength, 31);

                    DLOG(concat("blen: ", str((isCompressed ? 1 : -1) * blockLength), " poff: ", str(parseOffset)));
                    parseOffset += 4;
                    if (blockLength > (1<<22)) { FREE_ALL(log("Block length broken.")); break; }
                    aSet(compressedBlocks, blockCount++, string(ptr_t(parseOffset), ptr_t(parseOffset + blockLength) * (isCompressed ? 1 : -1)));
                    parseOffset += blockLength;
                    totalLen += blockLength;
                    if (blockCount == LZ4_BLOCK_COUNT) {
                        break;
                    }
                }
                if (parseOffset > ReadHeapOffset) {
                    fromIOPtr = ReadHeapOffset;
                    ptr_t len = ptr_t(parseOffset - int64_t(ReadHeapOffset));
                    DLOG(concat("supplemental read: ", str(ivec3(ReadOffset, ReadHeapOffset, len))));
                    readSync(filename, ReadOffset, len, string(ReadHeapOffset, ReadHeapOffset + len));
                }
                DLOG(concat("block count: ", str(blockCount)));
                if (blockCount >= LZ4_BLOCK_COUNT || ReadCount != bsz * BLOCK_COUNT || ReadHeapOffset+bsz*BLOCK_COUNT > readBuffer.y) {
                    DLOG("stop reading");
                    ReadOffset -= (ReadHeapOffset - parseOffset);
                    parseOffset = 0;
                    StopReading = 1;
                }

                i32heap[compressedBlocksCount] = blockCount;
                if (blockLength == 0) DLOG(concat("Total compressed length: ", str(totalLen)));
                ReadCount = (ReadCount == bsz*BLOCK_COUNT || blockCount == LZ4_BLOCK_COUNT) ? 0 : 1;

                IsLastBlock = blockCount == 0 ? 1 : 0;
                programReturnValue = 0;
            }
            while (programReturnValue != 0);
        }

        barrier(); memoryBarrier();
        blockCount = i32heap[compressedBlocksCount];

        int j = ThreadGroupId;
        while(IsLastBlock == 0) {
            for (int i = ThreadGroupId; j < blockCount && i < ThreadGroupCount; i += ThreadGroupCount, j += ThreadGroupCount) {
                string compressed = aGet(compressedBlocks,j);
                if (compressed.y < 0) { // Block without compression
                    compressed.y = abs(compressed.y);
                    parMemcpyFromHeapToHeap(compressed.x, readBuffer.y + i*(1<<22), strLen(compressed), ThreadLocalCount, ThreadLocalId);
                    if (ThreadLocalId == 0) i32heap[uncompressedLengths + i] = strLen(compressed);
                } else { // Compressed block
                    ptr_t writeEndPtr = lz4DecompressBlockFromHeapToHeap(compressed, string(readBuffer.y + (1<<22)*i, readBuffer.y + (1<<22)*(i+1)), LZ4Literals, LZ4Matches);
                    if (ThreadLocalId == 0) {
                        i32heap[uncompressedLengths + i] = writeEndPtr - (readBuffer.y + (i*(1<<22)));
                    }
                }
                barrier(); memoryBarrier();
            }
            atomicAdd(ReadBarrier, 1);
            if (ThreadId == 0) {
                while (ReadBarrier < ThreadCount);

                toIOPtr = 0;

                int64_t uncompressedLen = 0;
                int len = LZ4_BLOCK_COUNT;
                if (j > blockCount) {
                    len = blockCount % LZ4_BLOCK_COUNT;
                }
                for (int i = 0; i < len; i++) {
                    uncompressedLen += i32heap[uncompressedLengths + i];
                }
                DLOG(concat("Uncompressed ", str(len), " blocks to ", str(uncompressedLen)));
                totalUncompressedLen += uncompressedLen;

                WriteLength = int32_t(uncompressedLen);

                IsLastBlock = j >= blockCount ? 1 : 0;
                ReadBarrier = 0;
            }
            while (ReadBarrier != 0);
            barrier(); memoryBarrier();
            if (ThreadId < 16*32) for (int32_t i = ThreadId, l = (WriteLength+15)/16; i < l; i += 16*32) {
                i64v2toIO[i] = i64v2heap[i + readBuffer.y / 16];
            }
            barrier(); memoryBarrier();
            if (ThreadLocalId == 0) atomicAdd(ReadBarrier, 1);
            if (ThreadId == 0) {
                while (ReadBarrier < ThreadGroupCount);
                awaitIO(writeIO);
                writeIO = write(stdout, -1, size_t(WriteLength), string(0, ptr_t(WriteLength)), false);
                ReadBarrier = 0;
            }
            while (ReadBarrier != 0);
            barrier();
        }
        barrier(); memoryBarrier();
    }
    if (ThreadId == 0 && !firstBlock) {
        awaitIO(writeIO);
        DLOG(concat("Total uncompressed size: ", str(totalUncompressedLen)));
    }
}





    /*
        Refactor for performance:
        ===

        Goals:
        ---

        - [x] Process 256 blocks at a time
            - [x] Copy loaded blocks from fromIO to heap
            - [x] Parse compressed block offsets
            - [x] Stop when compressed block count is 256
            - [x] Rewind read offset to the end of block 256 (1 block takes as long to decompress as 72, so small tails can have perf impact)

        - [ ] Process 512 blocks at a time
            - [ ] Shift to using int64_t for pointers to do 512 blocks at a time (needs 4 GB ram for blocks)

        - [x] Improve block decompressor performance in lz4.glsl
            - [x] Make better use of local thread group
                - [x] Parse block to find literal offsets
                - [x] Splat literals at correct positions in parallel
                - [x] Run sequential match fill pass
            - [x] Use 4/8-byte copies where possible => No benefit
                - [x] Unaligned load
                - [x] Unaligned store
                - [x] Masked versions to do arbitrary byte length load/store

        - [ ] Group shuffles
            - [ ] Unaligned memcpy
            - [ ] Parse multiple bytes at a time (load a big chunk at a time, use shuffles to get bytes)
            - [ ] Use shuffle instructions to broadcast matches across thread group (no need for shared)

        - [ ] Faster match fill by filling a repeating prefix, then switching over to full-width memcpy

        - [ ] Instead of re-reading the tail of a block, memmove tail to start of block

        - [ ] Overlap read+parse, decompression and write
                     read .11 s | = 11 GB/s [125 GB/s effective]
            decompression .60 s | ====: 2 GB/s [22 GB/s effective]
                    write 2.2 s | ====:====:====:==== 6.2 GB/s


        Apply learnings to library design:
        ===

        Based on writing Grep, DeLZ4, PCIe LZ4 compression and IO experiments, how would you change or re-design the library?
        Write a network server at least, then figure out more.

        1. Shift to 64-bit pointers to use full memory capacity
        2. Make it easy to write stream processors, like UNIX pipes with a defined level of parallelism: (read[32] | parse)[256] | process[256] | write
        3. Helpers for global barriers, mutexed global arrays
        4. Memcpy between GPU-CPU and GPU-GPU have different optimal levels of parallelism (GPU-CPU = 16*32 threads copying 16 bytes at a time, GPU-GPU = lots of threads 16 bytes at a time)
        5. Expand IO ops in the IO layer (if read > 256kB, do it with X threads)
        6. Restartable program for long-running processes (decompressing 13 GB to disk sometimes takes longer than 10 seconds)
            void main() {
                if (RunCount == 0) {
                    ... do initial setup, store local vars into buffer
                } else {
                    ... load values from buffer to local vars
                }
                bool done = processOneStep();
                if (!done) RerunProgram = 1;
            }

        - [ ] IO library changes
            - Streaming writes
            - Scatter writes
            - Gather reads
            - Streaming reads
            - In-memory processing focus :?

        - [ ] Runtime design
            - [ ] Dealing with long-running processes
            - [ ] Helpers for storing and loading program state

        - [ ] Array library
            - [ ] Arrays on IO heaps

        - [ ] String library
            - [ ] Strings on IO heaps
            - [ ] Operator overloading for "foo" + "bar"
            - [ ] print macro that casts args to strings
            - [ ] printf :(

        - [ ] Concurrency primitives
            - [ ] Locks & barriers (device-wide barriers)
            - [ ] The multi-reader-multi-processor-multi-writer approach from the bottom of the file
            - [ ] Parallel reduction
            - [ ] Serial reduction
            - [ ] Contract scope of execution
            - [ ] Widen scope

        - [ ] Redo memory library design
            - [ ] Make it easier to use the lib correctly and fast (fromIO buffer overwrites, manual heap pointer manipulation, etc. ugh stuff)
                - [ ] Debug logs shouldn't screw up heap and ongoing IO processes
            - [ ] Fast single-thread & group & program -wide memcpy functions
            - [ ] Write pseudo-code version of program that works like you'd like it to
            - [ ] Non-string heap arrays
            - [ ] Port mimalloc to GLSL to have a proper malloc
            - 3 types of buffers:
                1. GPU heap (GPU-local, cached)
                2. toIO (GPU-local, volatile, accessible by CPU)
                3. fromIO (CPU-local, accessible by GPU)
            - Allocation usage differs
                - Heap allocs are more or less user-controlled => Not many surprises even when messing with heapPtr
                - IO allocs happen during IO function calls (copying filename to toIO, copying data to/from IO) => Need to be careful with IO calls after doing custom toIOPtr / fromIOPtr
                - Three types of buffer use:
                    1. Thread-local allocations
                    2. Group-local allocations
                    3. Global allocations
                - Now all are using the same memory => clashes when e.g. a part of global area is overwritten by one thread doing logging from its own heap slice
                    => Separate the heaps
                - Overflowing the heap causes silent corruption
                    => Heap guards in malloc
    */

    /*
    shared string compressed;
    shared int64_t compressionStatus;
    shared int32_t blockCount;

    const int32_t Empty = 0;
    const int32_t Accessing = 1;
    const int32_t Full = 2;


    if (ThreadGroupId == 0) { // IO ThreadGroup

        while (!done) {
            // Try to grab an available compressed block.
            for (int i = 0; i < BLOCK_COUNT; i++) {
                if (atomicCompSwap(compressedBlockAvailable[i], Empty, Accessing) == Empty) {
                    int32_t block = atomicAdd(blockCount, 1);
                    readIOs[i] = read(filename, blockOffsets[block], blockLengths[block]+4, compressedBlocks[i].buffer);
                    compressedBlocks[i].block = block;
                    compressedBlocks[i].offset = block * bsz;
                }
                if (atomicCompSwap(decompressedBlockAvailable[i], Full, Accessing) == Full) {
                    writeIOs[i] = write(outfile, decompressedBlocks[i].offset, decompressedBlocks[i].length, decompressedBlocks[i].data);
                }
                if (pollIO(readIOs[i])) {
                    compressedBlocks[i].data = awaitIO(readIOs[i], true);
                    blockOffsets[compressedBlocks[i].block+1] = blockOffsets[compressedBlocks[i].block] + blockLen;
                    compressedBlockAvailable[i] = Full;
                }
                if (pollIO(writeIOs[i])) {
                    awaitIO(writeIOs[i]);
                    decompressedBlockAvailable[i] = Empty;
                }
            }
        }

    } else { // Block decompression ThreadGroup

        while (!done) {
            // Try to grab an available compressed block.
            for (int i = 0; i < BLOCK_COUNT; i++) {
                if (atomicCompSwap(compressedBlockAvailable[i], Full, Accessing) == Full) {

                    // We have a compressed block!
                    // Acquire the decompressed block.
                    while(atomicCompSwap(decompressedBlockAvailable[i], Empty, Accessing) != Empty);

                    LZ4DecompressBlockFromHeapToHeap(compressedBlocks[i], decompressedBlocks[i]);

                    // Release the blocks.
                    decompressedBlocksAvailable[i] = Full;
                    compressedBlockAvailable[i] = Empty;
                }
            }
        }

    }
    */

/*

Pseudocode version to aid language design
===

Shape: {{#|-}#|-}

{ loop
    { loop (read 256 blocks of compressed data)
        # threadgroup parallel (read compressed data)
        | all threads (copy compressed data to GPU heap)
        - single thread (find block boundaries in compressed data)
    }
    # threadgroup parallel (decompress compressed blocks)
    | all threads (copy decompressed data to CPU)
    - single thread (write decompressed data to stdout)
}


alloc_t parseBlock(alloc_t readBuffer, inout LZ4Header header, out int64_t remaining) {
    uint32_t blockLength = readU32(readBuffer, header.parseOffset - header.readOffset);
    header.parseOffset += 4;
    alloc_t block = header.parseOffset + alloc_t(0, blockLength);
    header.parseOffset += blockLength;
    remaining = max(0, (header.parseOffset - header.readOffset) - strLen(readBuffer));
    return block;
}

allocArray readBlocks(File file, int blockCount, inout LZ4Header header) {
    global allocArray blocks = allocArrayWithSize(blockCount);
    global done = 0;
    for (int i = 0; i < blockCount && done == 0;) {
        alloc_t readBuffer = readSync(file, readSize);
        single {
            if (strLen(readBuffer) < readSize) done = 1;
            if (!header.initialized) header = parseLZ4Header(readBuffer);
            for (; i < blockCount; i++) {
                int64_t remaining = 0;
                alloc_t block = parseBlock(readBuffer, header, remaining);
                if (remaining > 0) {
                    alloc_t rest = readSync(file, remaining);
                    block = concat(block, rest);
                    break;
                }
                allocArrayPush(blocks, block);
                if (strLen(block) == 0) {
                    done = 1;
                    break;
                }
            }
        }
        globalBarrier();
    }
    return blocks;
}

alloc_t decompressBlocks(allocArray blocks, allocArray targets, LZ4Header header) {
    if (ThreadGroupId < arrLen(blocks)) {
        lz4DecompressBlock(aGet(blocks, ThreadGroupId), aGet(targets, ThreadGroupId), header);
    }
    globalBarrier();
}


LZ4Header header;
File file = openSync(filename, "r");

while (!file.eof) {
    allocArray blocks = readBlocks(file, 256, header);
    alloc_t decompressed = decompressBlocks(blocks, header);
    writeSync(stdout, decompressed);
}




int64_t readOffset = 0;
int64_t blockOffset = 0;
int64_t parseOffset = 0;
int64_t readSize = 72 Mi;
int64_t blockLength = 0;
bool blockReadDone = false;
int32_t blockCount = 0;
bool firstBlock = false;
const int32_t MaxBlockCount = 256;

while (!blockReadDone) {
    int64_t readLength = readSync(filename, readOffset, readSize, globalHeap + blockOffset);
    readOffset += readLength;
    processingDone = readLength < readSize;
    single {
        if (firstBlock) {
            parseOffset = readHeader(globalHeap + blockOffset);
            firstBlock = false;
        }
        while (parseOffset < readOffset) {
            blockLength = readBlockLength(parseOffset);
            parseOffset += 4;
            compressedBlocks[blockCount++] = parseOffset + heapSlice(0, blockLength);
            parseOffset += blockLength;
            if (blockCount == MaxBlockCount || blockLength == 0) {
                blockReadDone = true;
                break;
            }
        }
        if (parseOffset > readOffset) {
            `
        }
    }
}

*/

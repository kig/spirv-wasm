#!/usr/bin/env gls

ThreadLocalCount = 32;
ThreadGroupCount = 256;

TotalHeapSize =   1258291200;
TotalToIOSize =   83886080;
TotalFromIOSize = 83886080;

#define LZ4_GROUP_SIZE 32

#include <file.glsl>
#include <lz4.glsl>

#define BLOCK_COUNT 72
#define LZ4_BLOCK_COUNT 256

const int32_t bsz = (1<<20);

const int32_t compressedBlocksCount = ((LZ4_BLOCK_COUNT+1) * (1<<22)) / 4;
const stringArray compressedBlocks = (compressedBlocksCount + 1) + stringArray(0, 18000);

const int32_t uncompressedLengths = compressedBlocks.y;

const string readBuffer = string(0, BLOCK_COUNT*bsz);

#define IsLastBlock io_pad_5
#define ReadCount io_pad_4
#define ReadBarrier io_pad_3

void main() {

    string filename = aGet(argv, 1);

    int64_t readOffset = 0, parseOffset = 0;
    uint32_t blockLength = 0;

    LZ4FrameHeader header;

    int error;

    uint32_t totalLen = 0;
    int64_t totalUncompressedLen = 0;

    if (ThreadId == 0) {
        ReadCount = 0;
        ReadBarrier = 0;
        programReturnValue = 0;
    }
    while(programReturnValue != 0);
    bool firstBlock = true;
    
    barrier();

    uint32_t contentChecksum = 0;

    error = LZ4_OK;

    while (ReadCount == 0) {
        barrier(); memoryBarrier();
        if (ThreadLocalId == 0) {
            heapPtr = groupHeapStart;
            fromIOPtr = ThreadGroupId*bsz;
            toIOPtr = groupToIOStart;

            int readBlockCount = 0;
            io ios[BLOCK_COUNT];
            FREE_IO(
                for (int i=ThreadGroupId; i < BLOCK_COUNT; i+=ThreadGroupCount, fromIOPtr += (ThreadGroupCount-1)*bsz) {
                    ios[readBlockCount++] = read(filename, readOffset+int64_t(ThreadGroupId*bsz), bsz, string(i*bsz, (i+1)*bsz));
                    readOffset += bsz * min(BLOCK_COUNT, ThreadGroupCount);
                }
                for (int i = 0; i < readBlockCount; i++) {
                    string compressed = awaitIO(ios[i], true);
                    if (strLen(compressed) > 0) {
                        atomicAdd(ReadCount, 1);
                    }
                }
            )
        }
        atomicAdd(programReturnValue, 1);
        barrier();
        if (ThreadId == 0) {
            while (programReturnValue < ThreadCount);

            int32_t blockCount = 0;

            log(concat("reads done ", str(ReadCount), " / ", str(BLOCK_COUNT)));

            if (firstBlock) {
                parseOffset = readLZ4FrameHeaderFromIO(0, header, error);
                firstBlock = false;
                blockLength = 1;
            }
            while (parseOffset < int64_t(BLOCK_COUNT * bsz) && blockLength > 0) {
                blockLength = readU32fromIO(int32_t(parseOffset));
                bool isCompressed = !getBit(blockLength, 31);
                blockLength = unsetBit(blockLength, 31);

                //FREE(FREE_IO(log(concat("blen: ", str((isCompressed ? 1 : -1) * blockLength), " poff: ", str(parseOffset)))));
                parseOffset += 4;
                if (blockLength > (1<<22)) { log("Block length broken."); break; }
                aSet(compressedBlocks, blockCount++, string(parseOffset, int32_t(parseOffset + blockLength) * (isCompressed ? 1 : -1)));
                parseOffset += blockLength;
                totalLen += blockLength;
            }
            if (parseOffset > int64_t(BLOCK_COUNT*bsz)) {
                fromIOPtr = ptr_t(BLOCK_COUNT*bsz);
                size_t len = size_t(parseOffset - int64_t(BLOCK_COUNT*bsz));
                //FREE(FREE_IO(log(concat("fill read to: ", str(ivec2(fromIOPtr, len)), " readOffset: ", str(readOffset)))));
                awaitIO(read(filename, readOffset, len, string(fromIOPtr, fromIOPtr+len)), true);
            }

            i32heap[compressedBlocksCount] = blockCount;
            //FREE(FREE_IO(log(concat("block count: ", str(blockCount)))));
            //if (ReadCount != BLOCK_COUNT) FREE(FREE_IO(log(concat("Total compressed length: ", str(totalLen)))));
            if (blockLength == 0) FREE(FREE_IO(log(concat("Total compressed length: ", str(totalLen)))));
            ReadCount = ReadCount == BLOCK_COUNT ? 0 : 1;

            parseOffset = parseOffset % (BLOCK_COUNT * bsz);

            IsLastBlock = blockCount == 0 ? 1 : 0;
            programReturnValue = 0;
        }
        while (programReturnValue != 0);

        barrier();
        int32_t blockCount = i32heap[compressedBlocksCount];

        int j = ThreadId / LZ4_GROUP_SIZE;
        while(IsLastBlock == 0) {
        FREE(FREE_IO(
            for (int i = ThreadId / LZ4_GROUP_SIZE; j < blockCount && i < LZ4_BLOCK_COUNT; i += (ThreadCount / LZ4_GROUP_SIZE), j += (ThreadCount / LZ4_GROUP_SIZE)) {
//                continue;
                string compressed = aGet(compressedBlocks,j);
                if (compressed.y < 0) {
                    compressed.y = abs(compressed.y);
//                    parMemcpyFromIOToHeap(compressed.x, i*(1<<22), (1<<22), LZ4_GROUP_SIZE, ThreadId % LZ4_GROUP_SIZE);
                    parMemcpyFromIOToHeap(compressed.x, i*(1<<22), strLen(compressed), LZ4_GROUP_SIZE, ThreadId % LZ4_GROUP_SIZE);
                    if (ThreadId % LZ4_GROUP_SIZE == 0) i32heap[uncompressedLengths + i] = strLen(compressed);
                } else {
                    ptr_t writeEndPtr = lz4DecompressBlockFromIOToHeap(compressed, string(i*(1<<22), (i+1)*(1<<22)));
                    if (ThreadId % LZ4_GROUP_SIZE == 0) {
                        i32heap[uncompressedLengths + i] = writeEndPtr - (i*(1<<22));
                        //if ((1<<22) != i32heap[uncompressedLengths + i])
                        //   FREE(FREE_IO(eprintln(str( ivec4( j, blockCount, i*(1<<22), writeEndPtr-(i*(1<<22)) ) ))));
                        
                    }
                    //if (ThreadId % LZ4_GROUP_SIZE == 0) FREE(FREE_IO(eprintln(str( ivec4( j, blockCount, i*(1<<22), writeEndPtr-(i*(1<<22)) ) ))));
                }
            }
            atomicAdd(ReadBarrier, 1);
            barrier();
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
                //FREE(FREE_IO(log(concat("Uncompressed ", str(len), " blocks to ", str(uncompressedLen)))));
                totalUncompressedLen += uncompressedLen;

                IsLastBlock = j >= blockCount ? 1 : 0;
                ReadBarrier = 0;
            }
            while (ReadBarrier != 0);
            barrier();
        ))
        }
        barrier(); memoryBarrier();
    }
    if (ThreadId == 0) {
        log(concat("Total uncompressed size: ", str(totalUncompressedLen)));
    }
}






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


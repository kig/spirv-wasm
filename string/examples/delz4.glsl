#!/usr/bin/env gls

ThreadLocalCount = 128;
ThreadGroupCount = 1;

TotalHeapSize =   80000000;
TotalToIOSize =   80000000;
TotalFromIOSize = 80000000;

#define LZ4_GROUP_SIZE 16

#include <file.glsl>
#include <lz4.glsl>


// Continue decompression from where we left before.
// [frame hdr] [block] [block] [blo | ck] [block] [bloc | k] [block] [block] [end]

// [block 1 write target] [block 2 write target] [block 3 write target] ...

// If blocks are not independent, we need the previous uncompressed block for decompression.
// [previous uncompressed block] [current uncompressed write block]

// The max size of the compressed block is 4 + uncompressed block size.

// How this would work:
//
// 1. read frame header and first block from compressed
// 2. if (i + blockLength > strLen(compressed)) { memmove(compressed, i, blockLength-i); read(file, off, compressed, blockLength-i); }
// 3. decompress block from compressed into currentWriteBuffer
// 4. flip writeBuffers, shift compressed to start of buf, fill the compressed buf, continue
//
// For parallel decompression, would have to operate on multiple independent blocks at a time.
//
// 1. parallel read to load compressed data into global IO buffer
// 2. ThreadId 0 goes through loaded data, grabbing block offsets and lengths as it goes and storing them to the global heap
// 3. ThreadId 0 flips a global atomic to tell decompressors to get to work
// 3. Thread groups have heap allocation matching uncompressed block size
// 4. Each thread group uncompresses a block in parallel, then issues pwrite and toggles the block completion atomic
// 5. ThreadId 0 waits for block completions and replaces compressed blocks with new ones

// [6. for grep, issue text search on uncompressed blocks in file order]

#define BLOCK_COUNT 72

shared string compressed;
shared bool done;
shared int64_t compressionStatus;
shared string blocks[BLOCK_COUNT];
shared int32_t uncompressedLengths[BLOCK_COUNT];
shared int32_t blockCount;

const int32_t Empty = 0;
const int32_t Accessing = 1;
const int32_t Full = 2;

void main() {

    string filename = aGet(argv, 1);

    int32_t bsz = 1048576;

    string readBuffer = string(0, BLOCK_COUNT*(bsz+4));
    
    int64_t readOffset = 0;
    uint32_t blockLength = 0;

    LZ4FrameHeader header;

    int error;

    /*
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

    if (ThreadLocalId == 0) {
        io r = read(filename, 0, 20, readBuffer);
        awaitIO(r, true);
        done = false;
        readOffset = readLZ4FrameHeaderFromIO(readBuffer.x, header, error);
        blockLength = readI32fromIO(int32_t(readOffset));
        readOffset += 4;
        if (error != LZ4_OK) {
            done = true;
        }
    }
    barrier();

    uint32_t contentChecksum = 0;

    while (!done) {
        if (ThreadLocalId == 0) {
            heapPtr = 0;
            fromIOPtr = 0;
            toIOPtr = 0;

            int32_t bufOff = 0;
            blockCount = 0;
            io ios[BLOCK_COUNT];
            FREE_IO(
                for (int i = 0; i < BLOCK_COUNT; i++) {
                    /*
                    if (blockLength == 0) {
                        io hr = read(filename, readOffset, 24, string(i*bsz, (i+1)*bsz));
                        string tgt = awaitIO(hr, true);

                        if (strLen(tgt) >= 4 && header.hasContentChecksum) {
                            contentChecksum = readU32fromIO(tgt.x);
                            // Check previous frame content vs checksum (or not, as it may be)
                            eprintln(concat("Got content checksum ", str(contentChecksum)));
                            tgt.x += 4;
                        }
                        
                        ptr_t off = readLZ4FrameHeaderFromIO(tgt.x, header, error);
                        blockLength = readU32fromIO(off);
                        
                        eprintln(concat("Block length 0 at ", str(i), " off ", str(readOffset)));
                        eprintln(str(strLen(tgt)));
                        eprintln(str(off - i*bsz));
                        eprintln(str(blockLength));
                        eprintln(str(error));
                        readOffset += 4 + off - i*bsz;
                        
                        if (error != LZ4_OK) {
                            done = true;
                            break;
                        }
                    }
                    */
                    /*
                    io r = read(filename, readOffset, int32_t(blockLength)+4, string(i*bsz, (i+1)*bsz));
                    compressed = awaitIO(r, true);
                    if (strLen(compressed) < blockLength || blockLength == 0) {
                        done = true;
                        break;
                    }
                    readOffset += int64_t(strLen(compressed));
                    if (blockLength > 0) {
                        blocks[i] = slice(compressed, 0, -4);
                        blockCount++;
                    }
                    blockLength = readI32fromIO(compressed.y - 4);
                    //eprintln(str(blockLength));
                    */
                    ios[i] = read(filename, readOffset, bsz, string(i*bsz, (i+1)*bsz));
                    readOffset += bsz;
                }
                for (int i = 0; i < BLOCK_COUNT; i++) {
                    compressed = awaitIO(ios[i], true);
                    if (strLen(compressed) < bsz) {
                        done = true;
                        break;
                    }
                }
            )
        }

        barrier();

/*        
        int error = 0;
        if (ThreadLocalId < blockCount * LZ4_GROUP_SIZE) {
            int32_t i = ThreadLocalId / LZ4_GROUP_SIZE;
            ptr_t writeEndPtr = lz4DecompressBlockFromIOToHeap(blocks[i], string(i*bsz, (i+1)*bsz));
            uncompressedLengths[i] = writeEndPtr - i * bsz;
        }
        barrier();
*/

        if (ThreadLocalId == 0) {
            heapPtr = 0;
            fromIOPtr = 0;
            toIOPtr = 0;
            
            if (error != LZ4_OK) {
                eprintln(concat("Error decompressing LZ4: ", str(error)));
                setReturnValue(error);
                done = true;
            } else {
                for (int i = 0; i < blockCount; i++) {
                    //FREE(log(str(uncompressedLengths[i])));
                    //print(string(i*bsz, i*bsz + uncompressedLengths[i]));
                }
            }
        }
        barrier();
    }

}


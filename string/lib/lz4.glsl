#define LZ4_OK 0
#define LZ4_ERROR_MAGIC 1
#define LZ4_ERROR_VERSION 2
#define LZ4_ERROR_MISSING_CONTENT_CHECKSUM 3
#define LZ4_ERROR_DICT_NOT_SUPPORTED 4

void parMemcpyFromIOToHeap(ptr_t src, ptr_t dst, size_t len, size_t groupSize, ptr_t threadId) {

/*
    if ((src & 15) == 0 && (dst & 15) == 0) {
        len = (len+15) / 16;
        src /= 16;
        dst /= 16;
        int32_t k = 0;
        for (; k < len-(groupSize-1); k+=groupSize) i64v2heap[k+dst+threadId] = i64v2fromIO[k+src+threadId];
        if (k < len && threadId < len-k) i64v2heap[k+dst+threadId] = i64v2fromIO[k+src+threadId];
    } else if ((src & 7) == 0 && (dst & 7) == 0) {
        len = (len+7) / 8;
        src /= 8;
        dst /= 8;
        int32_t k = 0;
        for (; k < len-(groupSize-1); k+=groupSize) i64heap[k+dst+threadId] = i64fromIO[k+src+threadId];
        if (k < len && threadId < len-k) i64heap[k+dst+threadId] = i64fromIO[k+src+threadId];
    } else if ((src & 3) == 0 && (dst & 3) == 0) {
        len = (len+3) / 4;
        src /= 4;
        dst /= 4;
        int32_t k = 0;
        for (; k < len-(groupSize-1); k+=groupSize) i32heap[k+dst+threadId] = i32fromIO[k+src+threadId];
        if (k < len && threadId < len-k) i32heap[k+dst+threadId] = i32fromIO[k+src+threadId];
    } else 
*/
    {
        int32_t k = 0;
        for (; k < len-(groupSize-1); k+=groupSize) u8heap[k+dst+threadId] = u8fromIO[k+src+threadId];
        if (k < len && threadId < len-k) u8heap[k+dst+threadId] = u8fromIO[k+src+threadId];
    }
}

void parMemcpyFromHeapToHeap(ptr_t src, ptr_t dst, size_t len, size_t groupSize, ptr_t threadId) {
    int32_t k = 0;
    for (; k < len-(groupSize-1); k+=groupSize) u8heap[k+dst+threadId] = u8heap[k+src+threadId];
    if (k < len && threadId < len-k) u8heap[k+dst+threadId] = u8heap[k+src+threadId];
}

ptr_t lz4DecompressBlockFromIOToHeap(string cmp, string dst, string previousBlock) {
    int32_t subId = (ThreadLocalId % LZ4_GROUP_SIZE);

    ptr_t j = dst.x;

    for (ptr_t be=cmp.y, i=cmp.x, dbe=dst.y; i<be && j<dbe;) {
        uint8_t token = u8fromIO[i++];

        // Copy literal to heap
        int32_t litLen = (int32_t(token) >> 4) & 0xf;
        uint8_t c = uint8_t(litLen | (litLen << 4));
        int32_t matchLen = int32_t(token) & 0xf;
        while (c == 0xff) {
            c = u8fromIO[i++];
            litLen += int32_t(c);
        }
        parMemcpyFromIOToHeap(i, j, litLen, LZ4_GROUP_SIZE, subId);
        i += litLen;
        j += litLen;
        if (i == be) continue; // End of LZ4 chunk

        // Copy match to heap
        int32_t matchOff = (int32_t(u8fromIO[i  ]) << 0)
                         | (int32_t(u8fromIO[i+1]) << 8);
        i += 2;
        c = uint8_t(matchLen | (matchLen << 4));
        matchLen += 4;
        while (c == 0xff) {
            c = u8fromIO[i++];
            matchLen += int32_t(c);
        }
        int32_t maxSubSize = min(LZ4_GROUP_SIZE, matchOff);
        if (subId < maxSubSize) {
            int32_t matchStart = j - matchOff;
            if (matchStart < 0) {
                int32_t prevLen = -matchStart;
                parMemcpyFromHeapToHeap(strLen(previousBlock) - prevLen, j, prevLen, maxSubSize, subId);
                matchLen = max(0, matchLen - prevLen);
                matchStart = 0;
            }
            parMemcpyFromHeapToHeap(matchStart, j, matchLen, maxSubSize, subId);
        }
        j += matchLen;
    }
    return j;
}

ptr_t lz4DecompressBlockFromIOToHeap(string cmp, string dst) {
    int32_t subId = (ThreadLocalId % LZ4_GROUP_SIZE);

    ptr_t j = dst.x;

    for (ptr_t be=cmp.y, i=cmp.x, dbe=dst.y; i<be && j<dbe;) {
        uint8_t token = u8fromIO[i++];

        // Copy literal to heap
        int32_t litLen = (int32_t(token) >> 4) & 0xf;
        uint8_t c = uint8_t(litLen | (litLen << 4));
        int32_t matchLen = int32_t(token) & 0xf;
        while (c == 0xff) {
            c = u8fromIO[i++];
            litLen += int32_t(c);
        }
        parMemcpyFromIOToHeap(i, j, litLen, LZ4_GROUP_SIZE, subId);
        i += litLen;
        j += litLen;
        if (i == be) continue; // End of LZ4 chunk

        // Copy match to heap
        int32_t matchOff = (int32_t(u8fromIO[i  ]) << 0)
                         | (int32_t(u8fromIO[i+1]) << 8);
        i += 2;
        c = uint8_t(matchLen | (matchLen << 4));
        matchLen += 4;
        while (c == 0xff) {
            c = u8fromIO[i++];
            matchLen += int32_t(c);
        }
        int32_t maxSubSize = min(LZ4_GROUP_SIZE, matchOff);
        if (subId < maxSubSize) {
            parMemcpyFromHeapToHeap(j-matchOff, j, matchLen, maxSubSize, subId);
        }
        j += matchLen;
    }
    return j;
}

struct LZ4FrameHeader {
    uint64_t contentSize;
    uint32_t magic;
    uint32_t version;
    uint32_t dictId;
    size_t blockMaxSize;
    bool skippable;
    bool blockIndependence;
    bool hasBlockChecksum;
    bool hasContentSize;
    bool hasContentChecksum;
    bool hasDictId;
    uint8_t headerChecksum;
};

ptr_t readLZ4FrameHeaderFromIO(ptr_t i, out LZ4FrameHeader header, out int error) {
    uint32_t magic = readU32fromIO(i);
    header.magic = magic;
    i += 4;
    if (magic != 0x184D2204) { // Not an LZ4 frame
        if ((magic & 0xfffffff0) == 0x184D2A50) { // Skippable frame
            uint32_t frameSize = readU32fromIO(i);
            i += 4;
            header.contentSize = frameSize;
            header.skippable = true;
            return i;
        }
        error = LZ4_ERROR_MAGIC;
        return i-4;
    }

    uint8_t FLG = u8fromIO[i++];
    uint8_t version = FLG >> 6;
    if (version != uint8_t(1)) {
        error = LZ4_ERROR_VERSION;
        return i-1;
    }
    bool blockIndependenceFlag = getBit(FLG, 5);
    bool blockChecksumFlag = getBit(FLG, 4);
    bool contentSizeFlag = getBit(FLG, 3);
    bool contentChecksumFlag = getBit(FLG, 2);
    bool dictIdFlag = getBit(FLG, 0);

    header.blockIndependence = blockIndependenceFlag;
    header.hasBlockChecksum = blockChecksumFlag;
    header.hasContentSize = contentSizeFlag;
    header.hasContentChecksum = contentChecksumFlag;
    header.hasDictId = dictIdFlag;
    
    uint8_t BD = u8fromIO[i++];
    size_t blockMaxSize = 1 << (8 + 2*((BD & 0xff) >> 4));

    header.blockMaxSize = blockMaxSize;

    uint64_t contentSize = 0;
    uint32_t dictId = 0;

    if (contentSizeFlag) {
        contentSize = readU64fromIO(i);
        i += 8;
    }
    header.contentSize = contentSize;

    if (dictIdFlag) {
        dictId = readU32fromIO(i);
        i += 4;
        error = LZ4_ERROR_DICT_NOT_SUPPORTED;
        return i-4;
    }
    header.dictId = dictId;

    uint8_t headerChecksum = u8fromIO[i++];
    header.headerChecksum = headerChecksum;
    //log(str(uvec4(blockIndependenceFlag, blockChecksumFlag, contentSizeFlag, contentChecksumFlag)));
    //log(str(uvec4(blockMaxSize, contentSize, dictId, headerChecksum)));

    error = LZ4_OK;
    return i;
}

void lz4DecompressBlockStreamFromIOToHeap(int32_t blockIndex, int32_t blockSize, string cmp, string dst) {
    if (blockIndex >= 128) return;

    ptr_t i = cmp.x + 128 * 4;
    int32_t len = 0;
    for (int32_t b=0; b<=blockIndex && b<128; b++) {
        len = i32fromIO[cmp.x/4 + b];
        if (b == blockIndex || len <= 0) break;
        i += len;
    }
    if (len <= 0 || i >= cmp.y) return;
    ptr_t outputStart = dst.x + blockSize * blockIndex;
    lz4DecompressBlockFromIOToHeap(string(i, min(cmp.y, i+len)), string(outputStart, min(dst.y, outputStart + blockSize)));
}

ptr_t lz4DecompressFramesFromIOToHeap(string cmp, string dst, out int32_t error) {
    ptr_t i = cmp.x;
    ptr_t j = dst.x;
    LZ4FrameHeader header;
    while (i < cmp.y) {
        i = readLZ4FrameHeaderFromIO(i, header, error);
        if (error != LZ4_OK) return i;
        if (header.skippable) {
            i += ptr_t(header.contentSize);
            continue;
        }

        while (i < cmp.y) {
            uint32_t blockSize = readU32fromIO(i);
            //log(str(blockSize));
            i += 4;
            if (blockSize == 0) break;

            bool isCompressed = !getBit(blockSize, 31);
            blockSize = unsetBit(blockSize, 31);

            if (isCompressed) {
                j = lz4DecompressBlockFromIOToHeap(string(i, i+blockSize), string(j, dst.y));
            } else {
                copyFromIOToHeap(string(i, i + blockSize), string(j, j + blockSize));
                j += ptr_t(blockSize);
            }
            i += ptr_t(blockSize);

            uint32_t blockChecksum = 0;
            if (header.hasBlockChecksum) {
                blockChecksum = readU32fromIO(i);
                i += 4;
            }
        }

        if (header.hasContentChecksum) {
            if (i > cmp.y-4) {
                error = LZ4_ERROR_MISSING_CONTENT_CHECKSUM;
                return i;
            }
            uint32_t contentChecksum = readU32fromIO(i);
            i += 4;
            //log(str(contentChecksum));
        }
    }
    error = LZ4_OK;
    return j;
}

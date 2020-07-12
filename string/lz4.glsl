#define getBit(n, idx) (0 != ((n) & (1 << (idx))))
#define setBit(n, idx) ((n) | (1 << (idx)))
#define unsetBit(n, idx) ((n) & ~(1 << (idx)))

#define LZ4_OK 0
#define LZ4_ERROR_MAGIC 1
#define LZ4_ERROR_VERSION 2
#define LZ4_ERROR_MISSING_CONTENT_CHECKSUM 3
#define LZ4_ERROR_DICT_NOT_SUPPORTED 4

uint32_t readU32fromIO(ptr_t i) {
    return (
          (uint32_t(u8fromIO[i])   << 0u) 
        | (uint32_t(u8fromIO[i+1]) << 8u) 
        | (uint32_t(u8fromIO[i+2]) << 16u) 
        | (uint32_t(u8fromIO[i+3]) << 24u)
    );
}

uint64_t readU64fromIO(ptr_t i) {
    return packUint2x32(u32vec2(readU32fromIO(i), readU32fromIO(i+4)));
}



/*
i64vec4 rotateLeft(i64vec4 v, i64vec4 v2, int offset) {
    return (v << offset) | (i64vec4(v.yzw, v2.x) >> (64-offset));
}

i64vec4 rotateRight(i64vec4 v, i64vec4 v2, int offset) {
    return (i64vec4(v.w, v2.xyz) << (64-offset)) | (v2 >> offset);
}

i64vec4 rotateLeftBytes(i64vec4 v1, i64vec4 v2, int offset) {
    if (offset >= 24) {
        v1 = i64vec4(v1.w, v2.xyz);
        v2 = i64vec4(v2.w, 0, 0, 0);
    } else if (offset >= 16) {
        v1 = i64vec4(v1.zw, v2.xy);
        v2 = i64vec4(v2.zw, 0, 0);
    } else if (offset >= 8) {
        v1 = i64vec4(v1.yzw, v2.x);
        v2 = i64vec4(v2.yzw, 0);
    }
    return rotateLeft(v1, v2, (offset%8)*8);
}

i64vec4 rotateRightBytes(i64vec4 v1, i64vec4 v2, int offset) {
    if (offset >= 24) {
        v1 = i64vec4(v1.xyz, v2.x);
        v2 = i64vec4(v2.yzw, 0);
    } else if (offset >= 16) {
        v1 = i64vec4(v1.xy, v2.xy);
        v2 = i64vec4(v2.zw, 0, 0);
    } else if (offset >= 8) {
        v1 = i64vec4(v1.x, v2.xyz);
        v2 = i64vec4(v2.w, 0, 0, 0);
    }
    return rotateRight(v1, v2, (offset%8)*8);
}

i64vec4 unalignedLoad(ptr_t i) {
    int idx = i / 32;
    return rotateLeftBytes(i64v4fromIO[idx], i64v4fromIO[idx+1], i % 32);
}

void unalignedStore(ptr_t i, i64vec4 v2) {
    int idx = i / 32;
    i64v4heap[idx] = rotateRightBytes(i64v4heap[idx], v2, i % 32);
    i64v4heap[idx+1] = rotateRightBytes(v2, i64v4heap[idx+1], i % 32);
}
*/


ptr_t lz4DecompressBlockFromIOToHeap(string cmp, string dst) {
    int32_t subId = (ThreadLocalId % LZ4_GROUP_SIZE);

    ptr_t j = dst.x;

    for (ptr_t be=cmp.y, i=cmp.x, dbe=dst.y; i<be && j<dbe;) {
        uint8_t token = u8fromIO[i++];
        int32_t litLen = (int32_t(token) >> 4) & 0xf;
        uint8_t c = uint8_t(litLen | (litLen << 4));
        int32_t matchLen = int32_t(token) & 0xf;
        while (c == 0xff) {
            c = u8fromIO[i++];
            litLen += int32_t(c);
        }
        {
            int32_t k = 0;
            for (; k < litLen-(LZ4_GROUP_SIZE-1); k+=LZ4_GROUP_SIZE) u8heap[k+j+subId] = u8fromIO[k+i+subId];
            if (k < litLen && subId < litLen-k) u8heap[k+j+subId] = u8fromIO[k+i+subId];
            i += litLen;
            j += litLen;
        }

        if (((j-dst.x) & 8191) == 0 && matchLen == 0) { // End of LZ4 chunk
            continue;
        }

        int32_t matchOff = (int32_t(u8fromIO[i  ]) << 0)
                         | (int32_t(u8fromIO[i+1]) << 8);
        i += 2;
        c = uint8_t(matchLen | (matchLen << 4));
        matchLen += 4;
        while (c == 0xff) {
            c = u8fromIO[i++];
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
    return j;
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
    if (len <= 0 || i >= cmp.y) {
        return;
    }
    ptr_t outputStart = dst.x + blockSize * blockIndex;
    int32_t subId = (ThreadLocalId % LZ4_GROUP_SIZE);

    for (ptr_t be=min(cmp.y, i+len), j=outputStart, dbe=min(dst.y, outputStart+blockSize); i<be && j<dbe;) {
        uint8_t token = u8fromIO[i++];
        int32_t litLen = (int32_t(token) >> 4) & 0xf;
        uint8_t c = uint8_t(litLen | (litLen << 4));
        int32_t matchLen = int32_t(token) & 0xf;
        while (c == 0xff) {
            c = u8fromIO[i++];
            litLen += int32_t(c);
        }
        {
            int32_t k = 0;
            for (; k < litLen-(LZ4_GROUP_SIZE-1); k+=LZ4_GROUP_SIZE) u8heap[k+j+subId] = u8fromIO[k+i+subId];
            if (k < litLen && subId < litLen-k) u8heap[k+j+subId] = u8fromIO[k+i+subId];
            i += litLen;
            j += litLen;
        }

        if (((j-outputStart) & 8191) == 0 && matchLen == 0) { // End of LZ4 block
            continue;
        }

        int32_t matchOff = (int32_t(u8fromIO[i  ]) << 0)
                         | (int32_t(u8fromIO[i+1]) << 8);
        i += 2;
        c = uint8_t(matchLen | (matchLen << 4));
        matchLen += 4;
        while (c == 0xff) {
            c = u8fromIO[i++];
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

int lz4DecompressFramesFromIOToHeap(string cmp, string dst) {
    ptr_t i = cmp.x;
    ptr_t j = dst.x;
    while (i < cmp.y) {
        uint32_t magic = readU32fromIO(i);
        i += 4;
        if (magic != 0x184D2204) { // Not an LZ4 frame
            if ((magic & 0xfffffff0) == 0x184D2A50) { // Skippable frame
                uint32_t frameSize = readU32fromIO(i);
                i += 4;
                i += ptr_t(frameSize);
                continue;
            }
            return LZ4_ERROR_MAGIC;
        }

        uint8_t FLG = u8fromIO[i++];
        uint8_t version = FLG >> 6;
        if (version != uint8_t(1)) return LZ4_ERROR_VERSION;
        bool blockIndependenceFlag = getBit(FLG, 5);
        bool blockChecksumFlag = getBit(FLG, 4);
        bool contentSizeFlag = getBit(FLG, 3);
        bool contentChecksumFlag = getBit(FLG, 2);
        bool dictIdFlag = getBit(FLG, 0);
        
        uint8_t BD = u8fromIO[i++];
        int32_t blockMaxSize = 1 << (8 + 2*((BD & 0xff) >> 4));

        uint64_t contentSize = 0;
        uint32_t dictId = 0;

        if (contentSizeFlag) {
            contentSize = readU64fromIO(i);
            i += 8;
        }

        if (dictIdFlag) {
            dictId = readU32fromIO(i);
            i += 4;
            return LZ4_ERROR_DICT_NOT_SUPPORTED;
        }

        uint8_t headerChecksum = u8fromIO[i++];

        while (i < cmp.y) {
            uint32_t blockSize = readU32fromIO(i);
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
            if (blockChecksumFlag) {
                blockChecksum = readU32fromIO(i);
                i += 4;
            }
        }
        if (contentChecksumFlag) {
            if (i > cmp.y-4) {
                return LZ4_ERROR_MISSING_CONTENT_CHECKSUM;
            }
            uint32_t contentChecksum = readU32fromIO(i);
            i += 4;
        }
    }
    return j;
}


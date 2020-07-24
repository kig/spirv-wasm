#define getBit(n, idx) (0 != ((n) & (1 << (idx))))
#define setBit(n, idx) ((n) | (1 << (idx)))
#define unsetBit(n, idx) ((n) & ~(1 << (idx)))

int32_t readI32fromIO(ptr_t i) {
    return (
          (int32_t(u8fromIO[i])   << 0u)
        | (int32_t(u8fromIO[i+1]) << 8u)
        | (int32_t(u8fromIO[i+2]) << 16u)
        | (int32_t(u8fromIO[i+3]) << 24u)
    );
}

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

int32_t readI32heap(ptr_t i) {
    return (
          (int32_t(u8heap[i])   << 0u)
        | (int32_t(u8heap[i+1]) << 8u)
        | (int32_t(u8heap[i+2]) << 16u)
        | (int32_t(u8heap[i+3]) << 24u)
    );
}

uint32_t readU32heap(ptr_t i) {
    return (
          (uint32_t(u8heap[i])   << 0u)
        | (uint32_t(u8heap[i+1]) << 8u)
        | (uint32_t(u8heap[i+2]) << 16u)
        | (uint32_t(u8heap[i+3]) << 24u)
    );
}

uint64_t readU64heap(ptr_t i) {
    return packUint2x32(u32vec2(readU32heap(i), readU32heap(i+4)));
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

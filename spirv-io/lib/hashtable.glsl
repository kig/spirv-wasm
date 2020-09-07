// Based on https://github.com/nosferalatu/SimpleGPUHashTable/

#include <array.glsl>

struct i32map {
    alloc_t table;
    int32_t capacity;
    int32_t count;
};

// 32 bit Murmur3 hash
int32_t murmur3hash(int32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    if (k == -1) k = 0;
    return k;
}

/*T
    i32map ht = i32hAlloc(300);
    512 == ht.capacity;
    512*3 == strLen(ht.table);
    0 == ht.count;

    ht = i32hAlloc(256);
    256 == ht.capacity;
    256*3 == strLen(ht.table);
    0 == ht.count;

    ht = i32hAlloc(257);
    512 == ht.capacity;
    512*3 == strLen(ht.table);
    0 == ht.count;
*/
i32map i32hAlloc(int32_t size) {
    size = 1 << int32_t(ceil(log2(float(size))));
    i32map ht = i32map(malloc(4 * (int32_t(size) * 3), 4), size, 0);
    ht.table.x /= 4;
    ht.table.y /= 4;
    for (uint32_t i = ht.table.x; i < ht.table.y; i += 3) {
        i32heap[i] = -1;
        i32heap[i+1] = -1;
        i32heap[i+2] = 0;
    }
    return ht;
}

i32map i32hAlloc() {
    return i32hAlloc(16);
}

#define i32hIter(ht, key, value, body) { \
    for (uint32_t _i_ = ht.table.x; _i_ < ht.table.y; _i_ += 3) {\
        if (i32heap[_i_+1] != -1) {\
            key = i32heap[_i_];\
            value = i32heap[_i_+2];\
            body;\
        }\
    }\
}

#define f32hIter(ht, key, value, body) { \
    for (uint32_t _i_ = ht.table.x; _i_ < ht.table.y; _i_ += 3) {\
        if (i32heap[_i_+1] != -1) {\
            key = i32heap[_i_];\
            value = intBitsToFloat(i32heap[_i_+2]);\
            body;\
        }\
    }\
}

i32array i32hKeys(i32map ht) {
    heapPtr += 3 - (3 - (heapPtr & 3));
    ptr_t start = heapPtr;
    i32hIter(ht, int32_t k, int32_t v, {
        i32heap[heapPtr/4] = k;
        heapPtr += 4;
    })
    return i32array(start/4, heapPtr/4);
}

i32array f32hKeys(i32map ht) {
    return i32hKeys(ht);
}

/*T
    i32map ht = i32hAlloc(256);
    int32_t v = 0;

    i32hSet(ht, 45, 1);
    i32hSet(ht, 46, 2);
    i32hSet(ht, 47, 3);

    true == i32hGet(ht, 45, v);
    1 == v;

    i32hSet(ht, 45, 4);
    i32hSet(ht, 248, 5);

    true == i32hGet(ht, 46, v);
    2 == v;
    true == i32hGet(ht, 47, v);
    3 == v;
    true == i32hGet(ht, 45, v);
    4 == v;
    true == i32hGet(ht, 248, v);
    5 == v;

    256 == ht.capacity;

    log("Adding 260 keys");
    for (int32_t i = 0; i < 260; i++) {
        i32hSet(ht, i, i);
    }

    // Resized table
    512 == ht.capacity;

    log("Checking for keys");
    // Check if all the keys are still there
    for (int32_t i = 0; i < 260; i++) {
        true == i32hGet(ht, i, v);
        i == v;
    }

*/
void i32hSet(inout i32map ht, int32_t key, int32_t value) {
    if ((ht.count + 1) * 100 > ht.capacity * 70) {
        i32map nt = i32hAlloc(ht.capacity*2);
        //FREE_ALL( log(concat("Resize ", str(ivec2(ht.capacity, nt.capacity)))) );
        for (uint32_t i = ht.table.x; i < ht.table.y; i += 3) {
            if (i32heap[i+1] != -1) {
                int32_t idx = i32heap[i+1] & (nt.capacity-1);
                while (i32heap[nt.table.x + idx*3] != -1) {
                    idx = (idx + 1) & (nt.capacity-1);
                }
                i32heap[nt.table.x + idx*3    ] = i32heap[i];
                i32heap[nt.table.x + idx*3 + 1] = i32heap[i+1];
                i32heap[nt.table.x + idx*3 + 2] = i32heap[i+2];
                nt.count++;
            }
        }
        ht = nt;
    }
    int32_t h = murmur3hash(key);
    int32_t idx = h & (ht.capacity-1);
    while (i32heap[ht.table.x + idx*3] != -1 && i32heap[ht.table.x + idx*3] != key) {
        idx = (idx + 1) & (ht.capacity-1);
    }
    if (i32heap[ht.table.x + idx*3] == -1) ht.count++;
    i32heap[ht.table.x + idx*3] = key;
    i32heap[ht.table.x + idx*3 + 1] = h;
    i32heap[ht.table.x + idx*3 + 2] = value;
}

/*T
    i32map ht = i32hAlloc(256);
    int32_t v = 123;

    false == i32hGet(ht, 30, v);

    i32hSet(ht, 30, 321);

    true == i32hGet(ht, 30, v);
    321 == v;

    false == i32hGet(ht, 31, v);

    for (int32_t i = 32; i < 512; i++) {
        false == i32hGet(ht, i, v);
    }

*/
bool i32hGet(i32map ht, int32_t key, out int32_t value) {
    int32_t idx = murmur3hash(key) & (ht.capacity-1);
    while (true) {
        int32_t k = i32heap[ht.table.x + idx * 3];
        if (k == key) {
            int32_t kh = i32heap[ht.table.x + idx * 3 + 1];
            if (kh == -1) return false;
            value = i32heap[ht.table.x + idx * 3 + 2];
            return true;
        } else if (k == -1) {
            return false;
        }
        idx = (idx + 1) & (ht.capacity-1);
    }
    return false;
}

/*T
    i32map ht = i32hAlloc(256);
    int32_t v = 0;

    i32hSet(ht, 30, 321);

    true == i32hGet(ht, 30, v);
    321 == v;

    true == i32hDelete(ht, 30);

    false == i32hGet(ht, 30, v);

    i32hSet(ht, 30, 321);

    log("i32hDelete: Adding and deleting 468 keys");

    for (int32_t i = 32; i < 500; i++) {
        i32hSet(ht, i, i);
        true == i32hGet(ht, i, i);
        true == i32hDelete(ht, i);
    }

    log("i32hDelete: Checking that none of the keys exist");

    for (int32_t i = 32; i < 500; i++) {
        false == i32hGet(ht, i, v);
        false == i32hDelete(ht, i);
    }

    true == i32hGet(ht, 30, v);
    321 == v;

    log("i32hDelete: Check sequences of gets, sets and deletes");

    for (int32_t i = 0; i < 500; i+=3) {
        i32hSet(ht, i, i);
    }
    for (int32_t i = 0; i < 500; i+=7) {
        i32hDelete(ht, i);
    }
    for (int32_t i = 0; i < 500; i+=3) {
        if (i % 7 != 0) {
            true == i32hGet(ht, i, v);
            i == v;
            if (!i32hGet(ht, i, v)) {
                log(concat("err 1.1: ", str(i)));
            }
        } else {
            false == i32hGet(ht, i, v);
            if (i32hGet(ht, i, v)) {
                log(concat("err 1.2: ", str(i)));
            }
        }
    }

    for (int32_t i = 0; i < 500; i+=11) {
        i32hSet(ht, i, i);
    }
    for (int32_t i = 0; i < 500; i+=3) {
        i32hDelete(ht, i);
    }
    for (int32_t i = 0; i < 500; i+=11) {
        if (i % 3 != 0) {
            true == i32hGet(ht, i, v);
            i == v;
            if (!i32hGet(ht, i, v)) {
                log(concat("err 2.1: ", str(i)));
            }
        } else {
            false == i32hGet(ht, i, v);
            if (i32hGet(ht, i, v)) {
                log(concat("err 2.2: ", str(i)));
            }
        }
    }

*/
bool i32hDelete(inout i32map ht, int32_t key) {
    int32_t idx = murmur3hash(key) & (ht.capacity-1);
    while (true) {
        int32_t  k = i32heap[ht.table.x + idx * 3];
        if (k == key) {
            if (i32heap[ht.table.x + idx * 3 + 1] == -1) return false;
            i32heap[ht.table.x + idx * 3 + 1] = -1;
            return true;
        } else if (k == -1) {
            return false;
        }
        idx = (idx + 1) & (ht.capacity-1);
    }
    return false;
}

i32map f32hAlloc(int32_t size) {
    return i32hAlloc(size);
}

i32map f32hAlloc() {
    return f32hAlloc(16);
}

void f32hDelete(inout i32map ht, int32_t key) {
    i32hDelete(ht, key);
}

bool f32hGet(i32map ht, int32_t key, out float value) {
    int32_t v;
    bool rv = i32hGet(ht, key, v);
    value = intBitsToFloat(v);
    return rv;
}

void f32hSet(inout i32map ht, int32_t key, float value) {
    i32hSet(ht, key, floatBitsToInt(value));
}


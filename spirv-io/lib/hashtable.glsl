// Based on https://github.com/nosferalatu/SimpleGPUHashTable/


struct hashtable {
    alloc_t table;
    uint32_t capacity;
};

// 32 bit Murmur3 hash
uint32_t murmur3hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k;
}

/*T
    hashtable ht = allocHashtable(300);
    512*2 == strLen(ht.table);

    ht = allocHashtable(256);
    256*2 == strLen(ht.table);

    ht = allocHashtable(257);
    512*2 == strLen(ht.table);
*/
hashtable allocHashtable(uint32_t size) {
    size = 1 << uint32_t(ceil(log2(float(size))));
    hashtable ht = hashtable(malloc(4 * (int32_t(size) * 2 + 16), 4), size);
    ht.table /= 4;
    for (uint32_t i = ht.table.x; i < ht.table.y; i += 2) {
        i32heap[i] = -1;
        i32heap[i + 1] = 0;
    }
    ht.table.y -= 16;
    return ht;
}

/*T
    hashtable ht = allocHashtable(256);
    int32_t v = 0;

    hashSet(ht, 45, 1);
    hashSet(ht, 46, 2);
    hashSet(ht, 47, 3);

    true == hashGet(ht, 45, v);
    1 == v;

    hashSet(ht, 45, 4);
    hashSet(ht, 248, 5);

    true == hashGet(ht, 46, v);
    2 == v;
    true == hashGet(ht, 47, v);
    3 == v;
    true == hashGet(ht, 45, v);
    4 == v;
    true == hashGet(ht, 248, v);
    5 == v;

    log("Adding 260 keys");
    for (int32_t i = 0; i < 260; i++) {
        hashSet(ht, i, i);
    }

    // Resized table
    512*2 == strLen(ht.table);

    log("Checking for keys");
    // Check if all the keys are still there
    for (int32_t i = 0; i < 260; i++) {
        true == hashGet(ht, i, v);
        i == v;
    }

*/
void hashSet(inout hashtable ht, int32_t key, int32_t value) {
    uint32_t idx = murmur3hash(key) & (ht.capacity-1);
    uint32_t i = 0;
    while (i32heap[ht.table.x + idx*2] != -1 && i32heap[ht.table.x + idx*2] != key && i < 8) {
        idx++;
        i++;
    }
    if (i == 8) {
        hashtable nt = allocHashtable(ht.capacity*2);
        //FREE_ALL( log(concat("Resize ", str(ivec2(ht.capacity, nt.capacity)))) );
        for (i = ht.table.x; i < ht.table.y+16; i += 2) {
            if (i32heap[i] != -1) {
                uint32_t nidx = murmur3hash(i32heap[i]) & (nt.capacity-1);
                while (i32heap[nt.table.x + nidx*2] != -1) {
                    nidx++;
                }
                i32heap[nt.table.x + nidx*2] = i32heap[i];
                i32heap[nt.table.x + nidx*2 + 1] = i32heap[i+1];
            }
        }
        idx = murmur3hash(key) & (nt.capacity-1);
        while (i32heap[nt.table.x + idx*2] != -1 && i32heap[nt.table.x + idx*2] != key) {
            idx++;
        }
        ht = nt;
    }
    i32heap[ht.table.x + idx*2] = key;
    i32heap[ht.table.x + idx*2 + 1] = value;
}

/*T
    hashtable ht = allocHashtable(256);
    int32_t v = 123;

    false == hashGet(ht, 30, v);

    hashSet(ht, 30, 321);

    true == hashGet(ht, 30, v);
    321 == v;

    false == hashGet(ht, 31, v);

    for (int32_t i = 32; i < 512; i++) {
        false == hashGet(ht, i, v);
    }

*/
bool hashGet(hashtable ht, int32_t key, out int32_t value) {
    uint32_t idx = murmur3hash(key) & (ht.capacity-1);
    uint32_t i = 0;
    while (i < 8) {
        int32_t k = i32heap[ht.table.x + idx * 2];
        if (k == -1) return false;
        if (k == key) {
            value = i32heap[ht.table.x + idx * 2 + 1];
            return true;
        }
        idx++;
        i++;
    }
    return false;
}

/*T
    hashtable ht = allocHashtable(256);
    int32_t v = 0;

    hashSet(ht, 30, 321);

    true == hashGet(ht, 30, v);
    321 == v;

    true == hashDelete(ht, 30);

    false == hashGet(ht, 30, v);

    hashSet(ht, 30, 321);

    log("hashDelete: Adding and deleting 468 keys");

    for (int32_t i = 32; i < 500; i++) {
        hashSet(ht, i, i);
        true == hashGet(ht, i, i);
        true == hashDelete(ht, i);
    }

    log("hashDelete: Checking that none of the keys exist");

    for (int32_t i = 32; i < 500; i++) {
        false == hashGet(ht, i, v);
        false == hashDelete(ht, i);
    }

    true == hashGet(ht, 30, v);
    321 == v;
*/
bool hashDelete(hashtable ht, int32_t key) {
    uint32_t idx = murmur3hash(key) & (ht.capacity-1);
    uint32_t i = 0;
    while (i < 8) {
        int32_t k = i32heap[ht.table.x + idx * 2];
        if (k == -1) return false;
        if (k == key) {
            i32heap[ht.table.x + idx * 2] = -1;
            i32heap[ht.table.x + idx * 2 + 1] = 0;
            return true;
        }
        idx++;
        i++;
    }
    return false;
}

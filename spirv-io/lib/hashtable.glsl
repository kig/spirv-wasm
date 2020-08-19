// Based on https://github.com/nosferalatu/SimpleGPUHashTable/


struct hashtable {
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
    hashtable ht = allocHashtable(300);
    512 == ht.capacity;
    512*3 == strLen(ht.table);
    0 == ht.count;

    ht = allocHashtable(256);
    256 == ht.capacity;
    256*3 == strLen(ht.table);
    0 == ht.count;

    ht = allocHashtable(257);
    512 == ht.capacity;
    512*3 == strLen(ht.table);
    0 == ht.count;
*/
hashtable allocHashtable(int32_t size) {
    size = 1 << int32_t(ceil(log2(float(size))));
    hashtable ht = hashtable(malloc(4 * (int32_t(size) * 3), 4), size, 0);
    ht.table /= 4;
    for (uint32_t i = ht.table.x; i < ht.table.y; i += 3) {
        i32heap[i] = -1;
        i32heap[i+1] = -1;
        i32heap[i+2] = 0;
    }
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

    256 == ht.capacity;

    log("Adding 260 keys");
    for (int32_t i = 0; i < 260; i++) {
        hashSet(ht, i, i);
    }

    // Resized table
    512 == ht.capacity;

    log("Checking for keys");
    // Check if all the keys are still there
    for (int32_t i = 0; i < 260; i++) {
        true == hashGet(ht, i, v);
        i == v;
    }

*/
void hashSet(inout hashtable ht, int32_t key, int32_t value) {
    if ((ht.count + 1) * 100 > ht.capacity * 70) {
        hashtable nt = allocHashtable(ht.capacity*2);
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

    log("hashDelete: Check sequences of gets, sets and deletes");

    for (int32_t i = 0; i < 500; i+=3) {
        hashSet(ht, i, i);
    }
    for (int32_t i = 0; i < 500; i+=7) {
        hashDelete(ht, i);
    }
    for (int32_t i = 0; i < 500; i+=3) {
        if (i % 7 != 0) {
            true == hashGet(ht, i, v);
            i == v;
            if (!hashGet(ht, i, v)) {
                log(concat("err 1.1: ", str(i)));
            }
        } else {
            false == hashGet(ht, i, v);
            if (hashGet(ht, i, v)) {
                log(concat("err 1.2: ", str(i)));
            }
        }
    }

    for (int32_t i = 0; i < 500; i+=11) {
        hashSet(ht, i, i);
    }
    for (int32_t i = 0; i < 500; i+=3) {
        hashDelete(ht, i);
    }
    for (int32_t i = 0; i < 500; i+=11) {
        if (i % 3 != 0) {
            true == hashGet(ht, i, v);
            i == v;
            if (!hashGet(ht, i, v)) {
                log(concat("err 2.1: ", str(i)));
            }
        } else {
            false == hashGet(ht, i, v);
            if (hashGet(ht, i, v)) {
                log(concat("err 2.2: ", str(i)));
            }
        }
    }

*/
bool hashDelete(inout hashtable ht, int32_t key) {
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

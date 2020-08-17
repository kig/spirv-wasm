struct hashtable {
    alloc_t table;
}

int32_t hash(int32_t key) {
    return abs(key * 1473);
}

hashtable allocHashtable(int32_t size) {
    size = 1 << int32_t(ceil(log2(float(size))));
    alloc_t ht = hashtable(malloc(size * 2 + 8));
    return ht;
}

void hashSet(inout hashtable ht, int32_t key, int32_t value) {
    int32_t idx = hash(key) & ((strLen(ht.table)/2)-1);
    int32_t i = 0;
    while (i32heap[ht.table.x + idx*2] == -1 && i < 8) {
        idx++;
        i++;
    }
    if (i == 8) {
        alloc_t nt = hashtable(malloc(strLen(ht.table)*2 + 8));
        nt.table.y -= 8;
        for (i = nt.table.x; i < nt.table.y; i+=2) {
            i32heap[i] = -1;
        }
        for (i = ht.table.x; i < ht.table.y; i+=2) {
            if (i32heap[i] != -1) {
                int32_t nidx = hash(i32heap[i]) & ((strLen(nt.table)/2)-1);
                while (i32heap[nt.table.x + nidx*2] == -1) {
                    nidx++;
                }
                i32heap[nt.table.x + nidx*2] = i32heap[i];
                i32heap[nt.table.x + nidx*2 + 1] = i32heap[i+1];
            }
        }
        idx = hash(key) & ((strLen(nt.table)/2)-1);
        while (i32heap[nt.table.x + idx*2] == -1) {
            idx++;
        }
        ht = nt;
    }
    i32heap[ht.table.x + idx*2] = key;
    i32heap[ht.table.x + idx*2 + 1] = value;
}

bool hashGet(hashtable ht, int32_t key, out int32_t value) {
    int32_t idx = hash(key) & ((strLen(ht.table)/2)-1);
    int32_t i = 0;
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

void hashDelete(hashtable ht, int32_t key) {
    int32_t idx = hash(key) & ((strLen(ht.table)/2)-1);
    int32_t i = 0;
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

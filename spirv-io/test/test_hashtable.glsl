
#include <assert.glsl>
#include "../lib/hashtable.glsl"

ThreadLocalCount = 1;
ThreadGroupCount = 1;

HeapSize = 16777216;
ToIOSize = 16777216;
FromIOSize = 16777216;


void test_allocHashtable() {

    hashtable ht = allocHashtable(300);
    assert(512*2 == strLen(ht.table));
    ht = allocHashtable(256);
    assert(256*2 == strLen(ht.table));
    ht = allocHashtable(257);
    assert(512*2 == strLen(ht.table));
}

void test_hashSet() {

    hashtable ht = allocHashtable(256);
    int32_t v = 0;

    hashSet(ht, 45, 1);
    hashSet(ht, 46, 2);
    hashSet(ht, 47, 3);
    assert(true == hashGet(ht, 45, v));
    assert(1 == v);
    hashSet(ht, 45, 4);
    hashSet(ht, 248, 5);
    assert(true == hashGet(ht, 46, v));
    assert(2 == v);
    assert(true == hashGet(ht, 47, v));
    assert(3 == v);
    assert(true == hashGet(ht, 45, v));
    assert(4 == v);
    assert(true == hashGet(ht, 248, v));
    assert(5 == v);
    log("Adding 260 keys");
    for (int32_t i = 0; i < 260; i++) {
        hashSet(ht, i, i);
    }

    // Resized table
    assert(512*2 == strLen(ht.table));
    log("Checking for keys");
    // Check if all the keys are still there
    for (int32_t i = 0; i < 260; i++) {
    assert(true == hashGet(ht, i, v));
    assert(i == v);
    }


}

void test_hashGet() {

    hashtable ht = allocHashtable(256);
    int32_t v = 123;
    assert(false == hashGet(ht, 30, v));
    hashSet(ht, 30, 321);
    assert(true == hashGet(ht, 30, v));
    assert(321 == v);    assert(false == hashGet(ht, 31, v));
    for (int32_t i = 32; i < 512; i++) {
    assert(false == hashGet(ht, i, v));
    }


}

void test_hashDelete() {

    hashtable ht = allocHashtable(256);
    int32_t v = 0;

    hashSet(ht, 30, 321);
    assert(true == hashGet(ht, 30, v));
    assert(321 == v);    assert(true == hashDelete(ht, 30));    assert(false == hashGet(ht, 30, v));
    hashSet(ht, 30, 321);

    log("hashDelete: Adding and deleting 468 keys");

    for (int32_t i = 32; i < 500; i++) {
        hashSet(ht, i, i);
    assert(true == hashGet(ht, i, i));
    assert(true == hashDelete(ht, i));
    }

    log("hashDelete: Checking that none of the keys exist");

    for (int32_t i = 32; i < 500; i++) {
    assert(false == hashGet(ht, i, v));
    assert(false == hashDelete(ht, i));
    }
    assert(true == hashGet(ht, 30, v));
    assert(321 == v);
}

void main() {
    FREE_ALL(test_allocHashtable());
    FREE_ALL(test_hashSet());
    FREE_ALL(test_hashGet());
    FREE_ALL(test_hashDelete());
}

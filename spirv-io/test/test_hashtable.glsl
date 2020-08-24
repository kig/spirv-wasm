
#include <assert.glsl>
#include "../lib/hashtable.glsl"

ThreadLocalCount = 1;
ThreadGroupCount = 1;

HeapSize = 16777216;
ToIOSize = 16777216;
FromIOSize = 16777216;


void test_i32hAlloc() {

    i32map ht = i32hAlloc(300);
    assert(512 == ht.capacity);
    assert(512*3 == strLen(ht.table));
    assert(0 == ht.count);
    ht = i32hAlloc(256);
    assert(256 == ht.capacity);
    assert(256*3 == strLen(ht.table));
    assert(0 == ht.count);
    ht = i32hAlloc(257);
    assert(512 == ht.capacity);
    assert(512*3 == strLen(ht.table));
    assert(0 == ht.count);
}

void test_i32hSet() {

    i32map ht = i32hAlloc(256);
    int32_t v = 0;

    i32hSet(ht, 45, 1);
    i32hSet(ht, 46, 2);
    i32hSet(ht, 47, 3);
    assert(true == i32hGet(ht, 45, v));
    assert(1 == v);
    i32hSet(ht, 45, 4);
    i32hSet(ht, 248, 5);
    assert(true == i32hGet(ht, 46, v));
    assert(2 == v);
    assert(true == i32hGet(ht, 47, v));
    assert(3 == v);
    assert(true == i32hGet(ht, 45, v));
    assert(4 == v);
    assert(true == i32hGet(ht, 248, v));
    assert(5 == v);    assert(256 == ht.capacity);
    log("Adding 260 keys");
    for (int32_t i = 0; i < 260; i++) {
        i32hSet(ht, i, i);
    }

    // Resized table
    assert(512 == ht.capacity);
    log("Checking for keys");
    // Check if all the keys are still there
    for (int32_t i = 0; i < 260; i++) {
    assert(true == i32hGet(ht, i, v));
    assert(i == v);
    }


}

void test_i32hGet() {

    i32map ht = i32hAlloc(256);
    int32_t v = 123;
    assert(false == i32hGet(ht, 30, v));
    i32hSet(ht, 30, 321);
    assert(true == i32hGet(ht, 30, v));
    assert(321 == v);    assert(false == i32hGet(ht, 31, v));
    for (int32_t i = 32; i < 512; i++) {
    assert(false == i32hGet(ht, i, v));
    }


}

void test_i32hDelete() {

    i32map ht = i32hAlloc(256);
    int32_t v = 0;

    i32hSet(ht, 30, 321);
    assert(true == i32hGet(ht, 30, v));
    assert(321 == v);    assert(true == i32hDelete(ht, 30));    assert(false == i32hGet(ht, 30, v));
    i32hSet(ht, 30, 321);

    log("i32hDelete: Adding and deleting 468 keys");

    for (int32_t i = 32; i < 500; i++) {
        i32hSet(ht, i, i);
    assert(true == i32hGet(ht, i, i));
    assert(true == i32hDelete(ht, i));
    }

    log("i32hDelete: Checking that none of the keys exist");

    for (int32_t i = 32; i < 500; i++) {
    assert(false == i32hGet(ht, i, v));
    assert(false == i32hDelete(ht, i));
    }
    assert(true == i32hGet(ht, 30, v));
    assert(321 == v);
    log("i32hDelete: Check sequences of gets, sets and deletes");

    for (int32_t i = 0; i < 500; i+=3) {
        i32hSet(ht, i, i);
    }
    for (int32_t i = 0; i < 500; i+=7) {
        i32hDelete(ht, i);
    }
    for (int32_t i = 0; i < 500; i+=3) {
        if (i % 7 != 0) {
    assert(true == i32hGet(ht, i, v));
    assert(i == v);
            if (!i32hGet(ht, i, v)) {
                log(concat("err 1.1: ", str(i)));
            }
        } else {
    assert(false == i32hGet(ht, i, v));
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
    assert(true == i32hGet(ht, i, v));
    assert(i == v);
            if (!i32hGet(ht, i, v)) {
                log(concat("err 2.1: ", str(i)));
            }
        } else {
    assert(false == i32hGet(ht, i, v));
            if (i32hGet(ht, i, v)) {
                log(concat("err 2.2: ", str(i)));
            }
        }
    }


}

void main() {
    FREE_ALL(test_i32hAlloc());
    FREE_ALL(test_i32hSet());
    FREE_ALL(test_i32hGet());
    FREE_ALL(test_i32hDelete());
}

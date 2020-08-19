
#include <assert.glsl>
#include "../lib/array.glsl"

ThreadLocalCount = 1;
ThreadGroupCount = 1;

HeapSize = 16777216;
ToIOSize = 16777216;
FromIOSize = 16777216;


void test_i32alloc() {

    i32array a = i32alloc(3);
    assert(3 == i32len(a));
}

void test_i32len() {
    assert(3 == i32len(i32{1,2,3}));
    assert(1 == i32len(i32{1}));
    assert(0 == i32len(i32{}));
}

void test_i32get() {

    i32array a = i32{1,2,3};
    assert(1 == i32get(a, 0));
    assert(2 == i32get(a, 1));
    assert(3 == i32get(a, 2));
}

void test_i32set() {

    i32array a = i32{1,2,3};
    i32set(a, 0, 4);
    i32set(a, 1, 5);
    i32set(a, 2, 6);
    assert(4 == i32get(a, 0));
    assert(5 == i32get(a, 1));
    assert(6 == i32get(a, 2));
}

void test_i32last() {

    i32array a = i32{1,2,3};
    assert(3 == i32last(a));
}

void test_i32first() {

    i32array a = i32{1,2,3};
    assert(1 == i32first(a));
}

void test_i32clone() {

    i32array a = i32{1,2,3};
    i32array b = i32{4,5};
    assert(true == i32eq(i32clone(a), a));
    assert(true == i32eq(i32clone(b), b));
    assert(a.start != i32clone(a).start);
    assert(a.end != i32clone(a).end);
}

void test_i32concat() {

    i32array a = i32{1,2,3};
    i32array b = i32{4,5};
    i32array c = i32{};
    assert(true == i32eq(i32concat(a,b), i32{1,2,3,4,5}));
    assert(true == i32eq(i32concat(b,a), i32{4,5,1,2,3}));
    assert(true == i32eq(i32concat(a,c), i32{1,2,3}));
    assert(true == i32eq(i32concat(c,a), i32{1,2,3}));
    assert(true == i32eq(i32concat(c,c), i32{}));
}

void test_i32fill() {

    i32array a = i32{1,2,3,4,5};
    i32fill(a, 9);
    assert(true == i32eq(a, i32{9,9,9,9,9}));
}

void test_i32indexOf() {

    i32array a = i32{1,2,3,2,5};
    assert(0 == i32indexOf(a, 1));
    assert(4 == i32indexOf(a, 5));
    assert(1 == i32indexOf(a, 2));
    assert(2 == i32indexOf(a, 3));
    assert(-1 == i32indexOf(a, 4));
}

void test_i32lastIndexOf() {

    i32array a = i32{1,2,3,2,5};
    assert(0 == i32lastIndexOf(a, 1));
    assert(4 == i32lastIndexOf(a, 5));
    assert(3 == i32lastIndexOf(a, 2));
    assert(2 == i32lastIndexOf(a, 3));
    assert(-1 == i32lastIndexOf(a, 4));
}

void test_i32includes() {

    i32array a = i32{1,2,3,4,5};
    assert(true == i32includes(a, 1));
    assert(true == i32includes(a, 2));
    assert(true == i32includes(a, 3));
    assert(true == i32includes(a, 4));
    assert(true == i32includes(a, 5));
    assert(false == i32includes(a, 6));
}

void test_i32reverseInPlace() {

    i32array a = i32{1,2,3,4,5};
    i32reverseInPlace(a);
    assert(true == i32eq(a, i32{5,4,3,2,1}));
}

void test_i32reverse() {

    i32array a = i32{1,2,3,4,5};
    assert(true == i32eq(i32reverse(a), i32{5,4,3,2,1}));
}

void test_i32slice() {

    i32array a = i32{1,2,3,4,5};
    assert(3 == i32len(i32slice(a, 2)));
    assert(2 == i32len(i32slice(a, 3)));
    assert(2 == i32len(i32slice(a, -2)));
    assert(5 == i32len(i32slice(a, -5)));
    assert(0 == i32len(i32slice(a, 5)));
    assert(5 == i32len(i32slice(a, 0)));
    assert(3 == i32get(i32slice(a, 2), 0));
    // (i32array a, size_t i) => i32len(a) >= i32len(i32slice(a, i));

}

void test_i32slice_() {

    i32array arr = i32alloc(5);
    for (ptr_t i = 0; i < 5; i++) i32set(arr, i, i);
    assert(3 == i32len(i32slice(arr, 1, -1)));
    assert(3 == i32len(i32slice(arr, 1, 4)));
    assert(3 == i32len(i32slice(arr, 2, 7)));
    assert(3 == i32get(i32slice(arr, 1, -1), 2));
    // (i32array a, size_t i, size_t j) => i32len(a) >= i32len(i32slice(a, i, j));

}

void main() {
    FREE_ALL(test_i32alloc());
    FREE_ALL(test_i32len());
    FREE_ALL(test_i32get());
    FREE_ALL(test_i32set());
    FREE_ALL(test_i32last());
    FREE_ALL(test_i32first());
    FREE_ALL(test_i32clone());
    FREE_ALL(test_i32concat());
    FREE_ALL(test_i32fill());
    FREE_ALL(test_i32indexOf());
    FREE_ALL(test_i32lastIndexOf());
    FREE_ALL(test_i32includes());
    FREE_ALL(test_i32reverseInPlace());
    FREE_ALL(test_i32reverse());
    FREE_ALL(test_i32slice());
    FREE_ALL(test_i32slice_());
}

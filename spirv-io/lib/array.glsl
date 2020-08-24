struct i32array {
    ptr_t start;
    ptr_t end;
};


/*T
    i32array a = i32alloc(3);
    3 == i32len(a);
*/
i32array i32alloc(size_t size) {
    alloc_t a = malloc(size * 4, 4);
    return i32array(a.x/4, a.y/4);
}

/*T
    3 == i32len(i32{1,2,3});
    1 == i32len(i32{1});
    0 == i32len(i32{});
*/
size_t i32len(i32array arr) {
    return (arr.end - arr.start);
}

/*T
    i32array a = i32{1,2,3};
    1 == i32get(a, 0);
    2 == i32get(a, 1);
    3 == i32get(a, 2);
*/
int32_t i32get(i32array arr, size_t index) {
    if (index < 0 || index >= i32len(arr)) return 1<<32;
    return i32heap[arr.start + index];
}

/*T
    i32array a = i32{1,2,3};
    i32set(a, 0, 4);
    i32set(a, 1, 5);
    i32set(a, 2, 6);
    4 == i32get(a, 0);
    5 == i32get(a, 1);
    6 == i32get(a, 2);
*/
bool i32set(i32array arr, size_t index, int32_t value) {
    if (index < 0 || index >= i32len(arr)) return false;
    i32heap[arr.start + index] = value;
    return true;
}

bool i32eq(i32array a, i32array b) {
    if (i32len(a) != i32len(b)) return false;
    for (ptr_t i = a.start, j = b.start; i < a.end; i++, j++) {
        if (i32heap[i] != i32heap[j]) return false;
    }
    return true;
}

/*T
    i32array a = i32{1,2,3};
    3 == i32last(a);
*/
int32_t i32last(i32array arr) {
    return i32get(arr, i32len(arr)-1);
}

/*T
    i32array a = i32{1,2,3};
    1 == i32first(a);
*/
int32_t i32first(i32array arr) {
    return i32get(arr, 0);
}

/*T
    i32array a = i32{1,2,3};
    i32array b = i32{4,5};
    true == i32eq(i32clone(a), a);
    true == i32eq(i32clone(b), b);
    a.start != i32clone(a).start;
    a.end != i32clone(a).end;
*/
i32array i32clone(i32array a) {
    i32array b = i32alloc(i32len(a));
    for (ptr_t i = a.start, j = b.start; i < a.end; i++, j++) {
        i32heap[j] = i32heap[i];
    }
    return b;
}

/*T
    i32array a = i32{1,2,3};
    i32array b = i32{4,5};
    i32array c = i32{};
    true == i32eq(i32concat(a,b), i32{1,2,3,4,5});
    true == i32eq(i32concat(b,a), i32{4,5,1,2,3});
    true == i32eq(i32concat(a,c), i32{1,2,3});
    true == i32eq(i32concat(c,a), i32{1,2,3});
    true == i32eq(i32concat(c,c), i32{});
*/
i32array i32concat(i32array a, i32array b) {
    i32array c = i32alloc(i32len(a) + i32len(b));
    ptr_t p = c.start;
    for (ptr_t i = a.start; i < a.end; i++, p++) i32heap[p] = i32heap[i];
    for (ptr_t i = b.start; i < b.end; i++, p++) i32heap[p] = i32heap[i];
    return c;
}

/*B
setup:
    i32array a = i32alloc(1000000);
    i32array b = i32alloc(4);

Fill large array:
    i32fill(a, 9);

Fill small array:
    i32fill(b, 9);
*/
/*T
    i32array a = i32{1,2,3,4,5};
    i32fill(a, 9);
    true == i32eq(a, i32{9,9,9,9,9});
*/
void i32fill(i32array a, int32_t v) {
    for (ptr_t i = a.start; i < a.end; i++) i32heap[i] = v;
}

/*T
    i32array a = i32{1,2,3,2,5};
    0 == i32indexOf(a, 1);
    4 == i32indexOf(a, 5);
    1 == i32indexOf(a, 2);
    2 == i32indexOf(a, 3);
    -1 == i32indexOf(a, 4);
*/
ptr_t i32indexOf(i32array a, int32_t v) {
    for (ptr_t i = a.start; i < a.end; i++) {
        if (i32heap[i] == v) return i-a.start;
    }
    return -1;
}

/*T
    i32array a = i32{1,2,3,2,5};
    0 == i32lastIndexOf(a, 1);
    4 == i32lastIndexOf(a, 5);
    3 == i32lastIndexOf(a, 2);
    2 == i32lastIndexOf(a, 3);
    -1 == i32lastIndexOf(a, 4);
*/
ptr_t i32lastIndexOf(i32array a, int32_t v) {
    for (ptr_t i = a.end-1; i >= a.start; i--) {
        if (i32heap[i] == v) return i-a.start;
    }
    return -1;
}

/*T
    i32array a = i32{1,2,3,4,5};
    true == i32includes(a, 1);
    true == i32includes(a, 2);
    true == i32includes(a, 3);
    true == i32includes(a, 4);
    true == i32includes(a, 5);
    false == i32includes(a, 6);
*/
bool i32includes(i32array a, int32_t v) {
    return i32indexOf(a, v) != -1;
}

/*T
    i32array a = i32{1,2,3,4,5};
    i32reverseInPlace(a);
    true == i32eq(a, i32{5,4,3,2,1});
*/
void i32reverseInPlace(i32array a) {
    for (ptr_t i = a.start, j = a.end - 1; i < j; ++i, --j) {
        int32_t tmp = i32heap[i];
        i32heap[i] = i32heap[j];
        i32heap[j] = tmp;
    }
}

/*T
    i32array a = i32{1,2,3,4,5};
    true == i32eq(i32reverse(a), i32{5,4,3,2,1});
*/
i32array i32reverse(i32array a) {
    i32array b = i32alloc(i32len(a));
    for (ptr_t i = a.start, j = b.end-1; i < a.end; ++i, --j) {
        i32heap[j] = i32heap[i];
    }
    return b;
}

/*T
    i32array a = i32{1,2,3,4,5};
    3 == i32len(i32slice(a, 2));
    2 == i32len(i32slice(a, 3));
    2 == i32len(i32slice(a, -2));
    5 == i32len(i32slice(a, -5));
    0 == i32len(i32slice(a, 5));
    5 == i32len(i32slice(a, 0));
    3 == i32get(i32slice(a, 2), 0);
    // (i32array a, size_t i) => i32len(a) >= i32len(i32slice(a, i));
*/
i32array i32slice(i32array a, size_t start) {
    a.start += normalizeIndex(start, i32len(a));
    return a;
}

/*T
    i32array arr = i32alloc(5);
    for (ptr_t i = 0; i < 5; i++) i32set(arr, i, i);
    3 == i32len(i32slice(arr, 1, -1));
    3 == i32len(i32slice(arr, 1, 4));
    3 == i32len(i32slice(arr, 2, 7));
    3 == i32get(i32slice(arr, 1, -1), 2);
    // (i32array a, size_t i, size_t j) => i32len(a) >= i32len(i32slice(a, i, j));
*/
i32array i32slice(i32array a, size_t start, size_t end) {
    size_t len = i32len(a);
    ptr_t e = normalizeIndex(end, len);
    ptr_t s = min(e, normalizeIndex(start, len));
    return i32array(a.start + s, a.start + e);
}

#define SWAP_I32(i, j) { ptr_t _i = a.start+i, _j = a.start+j; int32_t s = min(i32heap[_i], i32heap[_j]), t = max(i32heap[_i], i32heap[_j]); i32heap[_i] = s; i32heap[_j] = t; }

void i32sort3(i32array a) {
    SWAP_I32(1, 2);SWAP_I32(0, 2);SWAP_I32(0, 1);
}

void i32sort4(i32array a) {
    SWAP_I32(0, 1);SWAP_I32(2, 3);SWAP_I32(0, 2);SWAP_I32(1, 3);
    SWAP_I32(1, 2);
}

void i32sort5(i32array a) {
    SWAP_I32(0, 1);SWAP_I32(3, 4);SWAP_I32(2, 4);SWAP_I32(2, 3);
    SWAP_I32(0, 3);SWAP_I32(0, 2);SWAP_I32(1, 4);SWAP_I32(1, 3);
    SWAP_I32(1, 2);
}

void i32sort6(i32array a) {
    SWAP_I32(1, 2);SWAP_I32(0, 2);SWAP_I32(0, 1);SWAP_I32(4, 5);
    SWAP_I32(3, 5);SWAP_I32(3, 4);SWAP_I32(0, 3);SWAP_I32(1, 4);
    SWAP_I32(2, 5);SWAP_I32(2, 4);SWAP_I32(1, 3);SWAP_I32(2, 3);
}

void i32sort7(i32array a) {
    SWAP_I32(1, 2);SWAP_I32(0, 2);SWAP_I32(0, 1);SWAP_I32(3, 4);
    SWAP_I32(5, 6);SWAP_I32(3, 5);SWAP_I32(4, 6);SWAP_I32(4, 5);
    SWAP_I32(0, 4);SWAP_I32(0, 3);SWAP_I32(1, 5);SWAP_I32(2, 6);
    SWAP_I32(2, 5);SWAP_I32(1, 3);SWAP_I32(2, 4);SWAP_I32(2, 3);
}

void i32sort8(i32array a) {
    SWAP_I32(0, 1);SWAP_I32(2, 3);SWAP_I32(0, 2);SWAP_I32(1, 3);
    SWAP_I32(1, 2);SWAP_I32(4, 5);SWAP_I32(6, 7);SWAP_I32(4, 6);
    SWAP_I32(5, 7);SWAP_I32(5, 6);SWAP_I32(0, 4);SWAP_I32(1, 5);
    SWAP_I32(1, 4);SWAP_I32(2, 6);SWAP_I32(3, 7);SWAP_I32(3, 6);
    SWAP_I32(2, 4);SWAP_I32(3, 5);SWAP_I32(3, 4);
}

void siftDown(i32array a, ptr_t lo, ptr_t hi, ptr_t first) {
    ptr_t root = lo;
    while (true) {
        ptr_t child = 2 * root + 1;
        if (child >= hi) break;
        if (child + 1 < hi && first+child < first+child+1) {
            child++;
        }
        if (!(first+root < first+child)) {
            return;
        }
        SWAP_I32(first+root, first+child);
        root = child;
    }
}

// heapSort from golang's sort.go
void i32sort(i32array a) {
    ptr_t first = 0;
    ptr_t lo = 0;
    ptr_t hi = a.end - a.start;
    if (hi <= 8) {
        // Use a sorting network for small arrays
        switch (hi) {
            case 8: i32sort8(a); break;
            case 7: i32sort7(a); break;
            case 6: i32sort6(a); break;
            case 5: i32sort5(a); break;
            case 4: i32sort4(a); break;
            case 3: i32sort3(a); break;
            case 2: SWAP_I32(0,1); break;
        }
        return;
    }
    for (ptr_t i = (hi - 1) / 2; i >= 0; i--) {
        siftDown(a, i, hi, first);
    }
    for (ptr_t i = hi - 1; i >= 0; i--) {
        SWAP_I32(first, first+i);
        siftDown(a, lo, i, first);
    }
}



/* Functional methods don't work in GLSL - either turn into macros or fuggeddaboutit

#define i32each(array, v, i, lambda) {i32array _array_ = array; for (ptr_t i = 0, l = i32en(_array_); i < l; i++) { int32_t v = i32heap[i + _array_.start]; lambda; }}
#define i32map(from, to, v, i, lambda) {i32array _to_ = to; i32each(from, v, i, { lambda; i32heap[i + _to_.start] = v; }); }

// i32each(myArr, v, i, i32et(myArr, i, v * 2));
// i32map(myArr, newArr, v, i, v = v * 2);

i32array i32ilter(i32array a, bool predicate(int32_t v, ptr_t i, i32array a)) {
    i32array res = i32lloc(i32en(a));
    ptr_t j = res.start;
    for (ptr_t i = a.start; i < a.end; i++) {
        if (predicate(i32heap[i], i-a.start, a)) i32heap[j++] = i32heap[i];
    }
    heapPtr = j * 4;
    res.end = j;
    return res;
}

i32option i32ind(i32array a, bool predicate(int32_t v, ptr_t i, i32array a)) {
    i32array res = i32lloc(i32en(a));
    ptr_t j = res.start;
    for (ptr_t i = a.start; i < a.end; i++) {
        if (predicate(i32heap[i], i-a.start, a)) return i32option(true, i32heap[i]);
    }
    return i32option(false, 0);
}

ptr_t i32indIndex(i32array a, bool predicate(int32_t v, ptr_t i, i32array a)) {
    i32array res = i32lloc(i32en(a));
    ptr_t j = res.start;
    for (ptr_t i = a.start; i < a.end; i++) {
        if (predicate(i32heap[i], i-a.start, a)) return i-a.start;
    }
    return -1;
}

i32array i32ap(i32array a, int32_t f(int32_t v, ptr_t i, i32array a)) {
    i32array res = i32lloc(i32en(a));
    for (ptr_t i = a.start, j = res.start; i < a.end; i++, j++) {
        i32heap[j] = f(i32heap[i], i-a.start, a);
    }
    return res;
}

void i32ach(i32array a, void f(int32_t v, ptr_t i, i32array a)) {
    for (ptr_t i = a.start; i < a.end; i++) {
        f(i32heap[i], i-a.start, a);
    }
    return res;
}

T reduce(i32array a, T f(T accum, int32_t v, ptr_t i, i32array a), T init) {
    for (ptr_t i = a.start; i < a.end; i++) {
        init = f(init, i32heap[i], i-a.start, a);
    }
    return init;
}


map(a, f) =
    let res = array(len(a))
    each(a, (x,i) => res[i] = f(x))
    res

filter(a, f) =
    let res = array(len(a))
    let j = 0
    each(a, (x,i) => if (f(x)) res[j++] = x)
    shrink(res, j)
    res

map(filter(a, p), f) =
    let res = array(len(a))
    let j = 0
    each(a, (x,i) => if (p(x)) res[j++] = f(x))
    shrink(res, j)
    res

*/


%%ARRAYGLOBALS%%

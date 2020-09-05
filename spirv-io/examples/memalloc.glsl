#include <file.glsl>

void main() {
    int64_t ptr;
    alloc_t res = malloc(8,8);
    awaitIO(memAlloc(100, res));
    ptr = i64heap[res.x/8];
    awaitIO(memWrite(ptr, "Hello, CPU!"));
    alloc_t buf = malloc(10);
    string s = awaitIO(memRead(ptr, buf));
    println(s);
    awaitIO(memFree(ptr));
}

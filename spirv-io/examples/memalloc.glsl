#include <file.glsl>

ThreadLocalCount = 4;
ThreadGroupCount = 4;

void main() {
    int64_t ptr;
    alloc_t res = malloc(8,8);

    awaitIO(memAlloc(100, res));
    ptr = i64heap[res.x/8];
    string hello = `Thread ${ThreadId} says: Hello, CPU!`;
    awaitIO(memWrite(ptr, hello));
    alloc_t buf = malloc(strLen(hello));
    string s = awaitIO(memRead(ptr, buf));
    println(s);
    awaitIO(memFree(ptr));

    if (ThreadId == 0) {
        // Allocate a 30 GB buffer
        awaitIO(memAlloc(30000000000L, res));
        ptr = i64heap[res.x/8];
        // Write something every 1 MB
        for (int64_t i = 0; i < 30000000000L; i+=1000000L) {
            FREE_ALL( awaitIO(memWrite(ptr + i, str(i))) );
        }
        // Test that the writes succeeded
        for (int64_t i = 0; i < 30000000000L; i+=1000000L) {
            FREE_ALL(
                string num = str(i);
                string rd = awaitIO(memRead( ptr + i, malloc(strLen(num)) ));
                if (!strEq(num, rd)) println(`Roundtrip failed at ${i}: ${num} != ${rd}`);
            )
        }
        println("Read-write roundtrips successful to a 30 GB buffer");
        awaitIO(memFree(ptr));
    }
}

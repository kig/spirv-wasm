
#include "file.glsl"

layout ( local_size_x = 32, local_size_y = 1, local_size_z = 1 ) in;

#define TEST(testFn) FREE(FREE_IO(printTest(testFn(), #testFn)))

void main() {
    initGlobals();

    alloc_t s = malloc(2048);
    awaitIO(_ioPingPong(s));

    if (ThreadID == 0) {
        println(concat("IO pingpong on ", str(ThreadCount), " threads, total bytes ", str(ThreadCount * 2048)));
    }
}


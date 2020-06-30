#define HEAP_SIZE 8192

#include "file.glsl"

layout ( local_size_x = 16, local_size_y = 1, local_size_z = 1 ) in;


bool testRead() {
    bool okShort = strEq(readSync("hello.txt", malloc(100)), "Hello, world!");

    string buf = malloc(100);
    int ok;
    io reqNum = read("hello.txt", 0, 100, buf);
    string res = awaitIO(reqNum, ok);
    bool okLong = strEq(res, "Hello, world!");

    return okShort && okLong;
}

bool testWrite() {
    string buf = malloc(100);
    string filename = concat("write", str(ThreadID), ".txt");

    awaitIO(createFile(filename));
    awaitIO(truncateFile(filename, 0));
    awaitIO(write(filename, 0, 100, "Write, write, write!"));
    bool firstOk = strEq(awaitIO(read(filename, 0, 100, buf)), "Write, write, write!");
    awaitIO(truncateFile(filename, 0));

    writeSync(filename, "Hello, world!");
    bool secondOk = strEq(readSync(filename, buf), "Hello, world!");
    awaitIO(truncateFile(filename, 0));
    awaitIO(deleteFile(filename));

    return firstOk && secondOk;
}

void printTest(bool ok, string name) {
    if (!ok || ThreadID == 0)
        println(concat(str(ThreadID), ": ", name, ok ? " successful" : " failed!"));
}

#define TEST(testFn) FREE(FREE_IO(printTest(testFn(), #testFn)))

void main() {
    initGlobals();

    TEST(testRead);
    TEST(testWrite);
}


#include "file.glsl"

bool testRead() {
    bool okShort = strEq(readSync("hello.txt", malloc(100)), "Hello, world!");

    string buf = malloc(100);
    int ok;
    int reqNum = read("hello.txt", 0, 100, buf);
    string res = awaitIO(reqNum, ok);
    bool okLong = strEq(res, "Hello, world!");

    return okShort && okLong;
}

bool testWrite() {
    string buf = malloc(100);
    string filename = concat("write", toString(gl_GlobalInvocationID.x), ".txt");

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
    if (!ok || gl_GlobalInvocationID.x == 0)
        print(concat(toString(gl_GlobalInvocationID.x), ":", name, ok ? " successful\n" : " failed!\n"));
}

#define TEST(testFn) printTest(testFn(), #testFn)

void main() {
    initGlobals();

    TEST(testRead);
    TEST(testWrite);
}


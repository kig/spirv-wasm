
#include "file.glsl"

layout ( local_size_x = 4, local_size_y = 1, local_size_z = 1 ) in;


bool testRead() {
    string r1 = readSync("hello.txt", malloc(100));
    bool okShort = strEq(r1, "Hello, world!");
    if (!okShort) println(concat(str(strLen(r1)), r1));

    string buf = malloc(100);
    int ok;
    io reqNum = read("hello.txt", 0, 100, buf);
    string res = awaitIO(reqNum, ok);
    bool okLong = strEq(res, "Hello, world!");
    if (!okLong) println(concat(str(strLen(res)), res));

    return okShort && okLong;
}

bool testWrite() {
    string buf = malloc(100);
    string filename = concat("write", str(ThreadID), ".txt");

    awaitIO(createFile(filename));
    awaitIO(truncateFile(filename, 0));
    awaitIO(write(filename, 0, 100, "Write, write, write!"));
    string r1 = awaitIO(read(filename, 0, 100, buf));
    bool firstOk = strEq(r1, "Write, write, write!");
    if (!firstOk) println(concat(str(strLen(r1)), r1));
    awaitIO(truncateFile(filename, 0));

    writeSync(filename, "Hello, world!");
    string r2 = readSync(filename, buf);
    bool secondOk = strEq(r2, "Hello, world!");
    if (!secondOk) println(concat(str(strLen(r2)), r2));
    awaitIO(truncateFile(filename, 0));
    awaitIO(deleteFile(filename));

    return firstOk && secondOk;
}

bool testRunCmd() {
    awaitIO(runCmd(concat("echo Hello from thread ", str(ThreadID))));
    return true;
}

#define rg(i,n) for (int i=0,_l_=(n); i<_l_; i++)
#define mapIO(i, n, f) { io _ios_[n]; rg(i, n) _ios_[i] = f; rg(i, n) awaitIO(_ios_[i]); }

bool testLs() {

    string dir = concat("dir-", str(ThreadID));
    awaitIO(mkdir(dir));

    mapIO(i, 10, createFile(concat(dir, "/", str(i))));

    stringArray res = awaitIO(ls(dir, malloc(1000)));

    mapIO(i, 10, deleteFile(concat(dir, "/", str(i))));

    awaitIO(rmdir(dir));

    bool ok = true;
    ok = ok && arrLen(res) == 10;
    rg(i, 10) {
        bool found = false;
        string si = str(i);
        rg(j, 10) {
            FREE(
                found = found || strEq(concat(dir, "/", si), aGet(res, j));
            )
        }
        ok = ok && found;
    }

    return ok;
}

void printTest(bool ok, string name) {
    if (!ok || ThreadID == 0) {
        println(concat(str(ThreadID), ": ", name, ok ? " successful" : " failed!"));
    }
}

#define TEST(testFn) FREE(FREE_IO(printTest(testFn(), #testFn)))

void main() {
    initGlobals();

    TEST(testRead);
    TEST(testWrite);
    TEST(testRunCmd);
    TEST(testLs);
}


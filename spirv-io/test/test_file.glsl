#!/usr/bin/env gls

#include "../lib/file.glsl"

ThreadLocalCount = 1;
ThreadGroupCount = 1;

#define rg(i,n) for (int i=0,_l_=(n); i<_l_; i++)
#define mapIO(i, n, f) { io _ios_[n]; rg(i, n) _ios_[i] = f; rg(i, n) awaitIO(_ios_[i]); }

bool testRead() {
    string r1 = readSync("hello.txt", malloc(100));
    bool okShort = strEq(r1, "Hello, world!\n");
    if (!okShort) println(concat(str(strLen(r1)), " ", r1));

    string buf = malloc(100);
    int ok;
    io reqNum = read("hello.txt", 0, 100, buf);
    string res = awaitIO(reqNum, ok);
    bool okLong = strEq(res, "Hello, world!\n");
    if (!okLong) println(concat(str(strLen(res)), " ", res));

    return okShort && okLong;
}

bool testWrite() {
    string buf = malloc(100);
    string filename = concat("write", str(ThreadId), ".txt");

    awaitIO(createFile(filename));
    awaitIO(truncateFile(filename, 0));
    awaitIO(write(filename, 0, 100, "Write, write, write!"));
    string r1 = readSync(filename, buf);
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
    FREE(FREE_IO(
        awaitIO(runCmd(concat("echo Hello from thread ", str(ThreadId))));
        awaitIO(runCmd(concat(
            "node -e 'fs=require(`fs`); fs.writeFileSync(`node-${",
            str(ThreadId),
            "}.txt`, Date.now().toString())'"
        )));
    ))
    string res = readSync(concat("node-", str(ThreadId), ".txt"), malloc(1000));
    println(concat("Node says ", res));
    deleteFile(concat("node-", str(ThreadId), ".txt"));
    return true;
}

bool testLs() {
    string dir = concat("dir-", str(ThreadId));
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

bool testGetCwd() {
    string cwd = awaitIO(getCwd());
    println(concat(str(ThreadId), " cwd is ", cwd));
    bool ok = strLen(cwd) > 0;
    if (ThreadId == 0) {
        awaitIO(mkdir("test_cwd"));
        awaitIO(chdir("test_cwd"));
        string newCwd = awaitIO(getCwd());
        println(concat("New cwd is ", newCwd));
        ok = ok && !strEq(cwd, newCwd);
        ok = ok && strEq("test_cwd", last(split(newCwd, '/')));
    }
    return ok;
}

void printTest(bool ok, string name) {
    if (!ok || ThreadId == 0) {
        println(concat(str(ThreadId), ": ", name, ok ? " successful" : " failed!"));
    }
}

#define TEST(testFn) FREE(FREE_IO(printTest(testFn(), #testFn)))

void main() {
    awaitIO(chdir("test_data"));
    TEST(testRead);
    TEST(testWrite);
    TEST(testRunCmd);
    TEST(testLs);
    TEST(testGetCwd);
}


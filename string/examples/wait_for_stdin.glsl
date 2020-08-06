#include <file.glsl>
#include <statemachine.glsl>

ThreadGroupCount = 1;
ThreadLocalCount = 1;

struct sio {
    io request;
    bool completed;
    string result;
};

bool storeIO(sio r, ptr_t idx) {
    ptr_t off = heapEnd/4 - (idx+1) * 5;
    if (off * 4 < heapStart) return false;
    i32heap[off] = r.request.index;
    i32heap[off+1] = r.request.heapBufStart;
    i32heap[off+2] = r.completed ? 0xffffffff : 0x80808080;
    i32heap[off+3] = r.result.x;
    i32heap[off+4] = r.result.y;
    return true;
}

bool loadIO(out sio r, ptr_t idx) {
    ptr_t off = heapEnd/4 - (idx+1) * 5;
    r.request = io(-1, -1);
    r.completed = false;
    r.result = string(0);
    if (i32heap[off+2] != 0x80808080 && i32heap[off+2] != 0xffffffff)
        return false;
    r.request.index = i32heap[off];
    r.request.heapBufStart = i32heap[off+1];
    r.completed = i32heap[off+2] == 0xffffffff;
    r.result.x = i32heap[off+3];
    r.result.y = i32heap[off+4];
    return true;
}

bool pollIO(inout sio r) {
    bool res = r.completed || pollIO(r.request);
    if (res && !r.completed) {
        r.result = awaitIO(r.request);
        r.completed = true;
    }
    return res;
}


const int s_Init = 0;
const int s_Reading = 1;

const int a_Read = 0;

void main() {
    /*
    Why not the way below?
    Because GPUs hang while waiting for IO and the driver kills the program after a few seconds.
    That said, making the below Just Work would be great.
        Compile awaitIO into "exit program with RERUN_ON_IO".
        Store and load IOs automatically.

    println("What's your name?");
    string name = awaitIO(readLine(stdin, malloc(256)));
    println(concat("Hello, ", name, "!"));
    */

    stateMachine m = loadStateMachine(s_Init);
    switch (getState(m)) {
        case s_Init:
            println("What's your name?");
            setAttr(m, a_Read, readLine(stdin, malloc(256)));
            setState(m, s_Reading);
            break;

        case s_Reading:
            io r = getIOAttr(m, a_Read);
            if (pollIO(r)) {
                string name = awaitIO(r);
                println(concat("Hello, ", name, "!"));
                return; // Done, exit program.
            }
            break;
    }
    saveStateMachine(m);
    rerunProgram = RERUN_ON_IO;

/*
    sio r;
    string res = malloc(256);
    if (!loadIO(r, 0)) {
        println("What's your name?");
        r.request = readLine(stdin, 256, res);
    }
    if (pollIO(r)) {
        println(concat("Hello, ", r.result, "!"));
    } else if (storeIO(r, 0)) {
        rerunProgram = RERUN_ON_IO;
    }
*/
}

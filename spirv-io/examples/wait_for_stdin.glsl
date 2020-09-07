#include <file.glsl>
#include <statemachine.glsl>

ThreadGroupCount = 1;
ThreadLocalCount = 1;

const int s_Init = 0;
const int s_Reading = 1;

const int a_Read = 0;

void main() {
    /*
    Why not the easy way[1]?

    Because GPUs hang while waiting for IO and the driver kills the program after a few seconds.
    That said, making the below Just Work would be great.
        Compile awaitIO into "exit program with RERUN_ON_IO".
        Store and load IOs automatically.

    [1] The easy way
    println("What's your name?");
    string name = awaitIO(readLine(stdin, malloc(256)));
    println(concat("Hello, ", name, "!"));
    */

    stateMachine m = loadStateMachine(s_Init);
    rerunProgram = RERUN_ON_IO;
    switch (getState(m)) {
        case s_Init:
            println("What's your name?");
            setAttr(m, a_Read, readLine(stdin, malloc(256)));
            setState(m, s_Reading);
            break;

        case s_Reading:
            io r = getIOAttr(m, a_Read);
            if (pollIO(r)) {
                rerunProgram = NO_RERUN;
                string name = awaitIO(r);
                println(concat("Hello, ", name, "!"));
                return; // Done, exit program.
            }
            break;
    }
    saveStateMachine(m);
}

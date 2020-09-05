#include <file.glsl>
ThreadLocalCount = 8;
ThreadGroupCount = 1;

void main() {
    for (int i = 0; i < 10; i++) {
        if (i > 3 && ThreadId > 2) stopIO = 1;
        if (ThreadId == 0 && i > 6) exitSync(0);
        println(str(ThreadId), ": ", str(i));
    }
}


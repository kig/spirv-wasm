#include <file.glsl>

void main() {
    if (ThreadId == 0) {
        if (runCount < 10) {
            println(concat("Hello from run ", str(runCount)));
            rerunProgram = RERUN_NOW;
        }
    }
}

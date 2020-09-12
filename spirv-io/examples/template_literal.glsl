#include <file.glsl>

ThreadLocalCount = 1;
ThreadGroupCount = 1;

void main() {
    float x = 1.2345;
    string s = `x = ${x}
x * x = ${
    x * x
}
Hello template literal!`;

    println(s);
}

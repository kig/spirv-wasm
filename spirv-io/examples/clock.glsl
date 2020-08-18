#include <file.glsl>

ThreadGroupCount = 1;
ThreadLocalCount = 1;

void main() {
    uint64_t ct = clockARB();
    uint64_t rt = clockRealtimeEXT();
    int64_t t0 = microTimeSync();
    int64_t t1 = microTimeSync();
    int64_t t2 = microTimeSync();
    int64_t t3 = microTimeSync();
    int64_t t4 = microTimeSync();
    uint64_t ct2 = clockARB();
    uint64_t rt2 = clockRealtimeEXT();

    println(concat("Wallclock time: ", str(t0)));
    println(concat("Wallclock time: ", str(t1)));
    println(concat("Wallclock time: ", str(t2)));
    println(concat("Wallclock time: ", str(t3)));
    println(concat("Wallclock time: ", str(t4)));
    println(concat("clock: ", str(ct), " elapsed ", str(ct2-ct)));
    println(concat("clockRealtime: ", str(rt), " elapsed ", str(rt2-rt)));
}

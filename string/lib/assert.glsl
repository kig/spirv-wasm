#include <file.glsl>

#define assert(f) { if (!(f)) { FREE_ALL(eprintln(concat(__FILE__, ":", str(__LINE__), " Assertion failed: ", #f ))); } }

#include <thread_id.glsl>

#define ptr_t int32_t
#define size_t int32_t

struct alloc_t { ptr_t x; ptr_t y; };
struct pair_t { alloc_t x; alloc_t y; };

#define INDEX_SIZE 4

#define FREE(f) { int32_t _hp_ = heapPtr; f; heapPtr = _hp_; }

layout(std430, binding = 0) buffer indexBuf { ptr_t indexHeap[]; };

layout(std430, binding = 0) buffer f32Buffer { float32_t f32heap[]; };
layout(std430, binding = 0) buffer f64Buffer { float64_t f64heap[]; };

layout(std430, binding = 0) buffer u8Buffer { uint8_t u8heap[]; };
layout(std430, binding = 0) buffer u16Buffer { uint16_t u16heap[]; };
layout(std430, binding = 0) buffer u32Buffer { uint32_t u32heap[]; };
layout(std430, binding = 0) buffer u64Buffer { uint64_t u64heap[]; };

layout(std430, binding = 0) buffer i8Buffer { int8_t i8heap[]; };
layout(std430, binding = 0) buffer i16Buffer { int16_t i16heap[]; };
layout(std430, binding = 0) buffer i32Buffer { int32_t i32heap[]; };
layout(std430, binding = 0) buffer i64Buffer { int64_t i64heap[]; };

#ifdef FLOAT16
layout(std430, binding = 0) buffer f16Buffer { float16_t f16heap[]; };
layout(std430, binding = 0) buffer f16v2Buffer { f16vec2 f16v2heap[]; };
layout(std430, binding = 0) buffer f16v3Buffer { f16vec3 f16v3heap[]; };
layout(std430, binding = 0) buffer f16v4Buffer { f16vec4 f16v4heap[]; };
layout(std430, binding = 0) buffer f16m2Buffer { f16mat2 f16m2heap[]; };
layout(std430, binding = 0) buffer f16m3Buffer { f16mat3 f16m3heap[]; };
layout(std430, binding = 0) buffer f16m4Buffer { f16mat4 f16m4heap[]; };
#endif

layout(std430, binding = 0) buffer f32v2Buffer { f32vec2 f32v2heap[]; };
layout(std430, binding = 0) buffer f64v2Buffer { f64vec2 f64v2heap[]; };

layout(std430, binding = 0) buffer f32v3Buffer { f32vec3 f32v3heap[]; };
layout(std430, binding = 0) buffer f64v3Buffer { f64vec3 f64v3heap[]; };

layout(std430, binding = 0) buffer f32v4Buffer { f32vec4 f32v4heap[]; };
layout(std430, binding = 0) buffer f64v4Buffer { f64vec4 f64v4heap[]; };

layout(std430, binding = 0) buffer u8v2Buffer { u8vec2 u8v2heap[]; };
layout(std430, binding = 0) buffer u16v2Buffer { u16vec2 u16v2heap[]; };
layout(std430, binding = 0) buffer u32v2Buffer { u32vec2 u32v2heap[]; };
layout(std430, binding = 0) buffer u64v2Buffer { u64vec2 u64v2heap[]; };

layout(std430, binding = 0) buffer u8v3Buffer { u8vec3 u8v3heap[]; };
layout(std430, binding = 0) buffer u16v3Buffer { u16vec3 u16v3heap[]; };
layout(std430, binding = 0) buffer u32v3Buffer { u32vec3 u32v3heap[]; };
layout(std430, binding = 0) buffer u64v3Buffer { u64vec3 u64v3heap[]; };

layout(std430, binding = 0) buffer u8v4Buffer { u8vec4 u8v4heap[]; };
layout(std430, binding = 0) buffer u16v4Buffer { u16vec4 u16v4heap[]; };
layout(std430, binding = 0) buffer u32v4Buffer { u32vec4 u32v4heap[]; };
layout(std430, binding = 0) buffer u64v4Buffer { u64vec4 u64v4heap[]; };

layout(std430, binding = 0) buffer i8v2Buffer { i8vec2 i8v2heap[]; };
layout(std430, binding = 0) buffer i16v2Buffer { i16vec2 i16v2heap[]; };
layout(std430, binding = 0) buffer i32v2Buffer { i32vec2 i32v2heap[]; };
layout(std430, binding = 0) buffer i64v2Buffer { i64vec2 i64v2heap[]; };

layout(std430, binding = 0) buffer i8v3Buffer { i8vec3 i8v3heap[]; };
layout(std430, binding = 0) buffer i16v3Buffer { i16vec3 i16v3heap[]; };
layout(std430, binding = 0) buffer i32v3Buffer { i32vec3 i32v3heap[]; };
layout(std430, binding = 0) buffer i64v3Buffer { i64vec3 i64v3heap[]; };

layout(std430, binding = 0) buffer i8v4Buffer { i8vec4 i8v4heap[]; };
layout(std430, binding = 0) buffer i16v4Buffer { i16vec4 i16v4heap[]; };
layout(std430, binding = 0) buffer i32v4Buffer { i32vec4 i32v4heap[]; };
layout(std430, binding = 0) buffer i64v4Buffer { i64vec4 i64v4heap[]; };

layout(std430, binding = 0) buffer f32m2Buffer { f32mat2 f32m2heap[]; };
layout(std430, binding = 0) buffer f64m2Buffer { f64mat2 f64m2heap[]; };

layout(std430, binding = 0) buffer f32m3Buffer { f32mat3 f32m3heap[]; };
layout(std430, binding = 0) buffer f64m3Buffer { f64mat3 f64m3heap[]; };

layout(std430, binding = 0) buffer f32m4Buffer { f32mat4 f32m4heap[]; };
layout(std430, binding = 0) buffer f64m4Buffer { f64mat4 f64m4heap[]; };


ptr_t heapStart = ThreadId * HeapSize;
ptr_t heapEnd = heapStart + HeapSize;

ptr_t heapPtr = heapStart;

ptr_t groupHeapStart = ThreadGroupId * GroupHeapSize;
ptr_t groupHeapEnd = groupHeapStart + GroupHeapSize;

shared ptr_t groupHeapPtr;

size_t allocSize(alloc_t a) {
	return a.y - a.x;
}

alloc_t malloc(size_t len) {
	ptr_t ptr = heapPtr;
	heapPtr += len;
	return alloc_t(ptr, heapPtr);
}

alloc_t malloc(size_t len, size_t align) {
	ptr_t ptr = ((heapPtr+(align-1)) / align) * align;
	heapPtr = ptr + len;
	return alloc_t(ptr, heapPtr);
}

alloc_t malloc(uint64_t len) {
	ptr_t ptr = heapPtr;
	heapPtr += ptr_t(len);
	return alloc_t(ptr, heapPtr);
}

alloc_t malloc(uint64_t len, size_t align) {
	ptr_t ptr = ((heapPtr+(align-1)) / align) * align;
	heapPtr = ptr + ptr_t(len);
	return alloc_t(ptr, heapPtr);
}

alloc_t malloc(uint32_t len) {
	ptr_t ptr = heapPtr;
	heapPtr += ptr_t(len);
	return alloc_t(ptr, heapPtr);
}

alloc_t malloc(uint32_t len, size_t align) {
	ptr_t ptr = ((heapPtr+(align-1)) / align) * align;
	heapPtr = ptr + ptr_t(len);
	return alloc_t(ptr, heapPtr);
}

ptr_t toIndexPtr(ptr_t ptr) {
    return ((ptr+(INDEX_SIZE-1)) / INDEX_SIZE);
}

ptr_t fromIndexPtr(ptr_t ptr) {
    return ptr * INDEX_SIZE;
}

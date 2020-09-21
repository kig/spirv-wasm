/*
 * Copyright 2015-2017 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#include <unistd.h>
#endif

#include "spirv_cross/external_interface.h"
#include "spirv_cross/internal_interface.hpp"
#include <stdio.h>

#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <map>
#include <atomic>
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <zstd.h>
#include <lz4.h>
#include <lz4frame.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <malloc.h>

#include "../lib/io.glsl"

#ifdef WIN32
#include <io.h>
#include <fcntl.h>
#endif

#define NDEBUG

#ifndef GLM_FORCE_SWIZZLE
#define GLM_FORCE_SWIZZLE
#endif

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/glm.hpp>
using namespace glm;

struct ioRequest {
    int32_t ioType;
    int32_t status;
    int64_t offset;
    int64_t count;
    int32_t filename_start;
    int32_t filename_end;
    int32_t result_start;
    int32_t result_end;
    int32_t compression;
    int32_t progress;
    int32_t data2_start;
    int32_t data2_end;
    int32_t _pad14;
    int32_t _pad15;
}; // 64 bytes

#ifndef IO_REQUEST_COUNT
#define IO_REQUEST_COUNT 65535
#endif

struct ioRequests {
    int32_t ioCount;
    int32_t programReturnValue;
    int32_t maxIOCount;
    int32_t runCount;
    int32_t rerunProgram;
    int32_t _pad5;
    int32_t _pad6;
    int32_t _pad7;
    int32_t _pad8;
    int32_t _pad9;
    int32_t _pad10;
    int32_t _pad11;
    int32_t _pad12;
    int32_t _pad13;
    int32_t _pad14;
    int32_t _pad15;
    ioRequest requests[IO_REQUEST_COUNT];
};


static std::mutex completed_mutex;
static std::mutex fileCache_mutex;
static std::atomic<int64_t> IOProgressCount(0);

class ComputeApplication
{
  protected:
    uint32_t heapBufferSize = 0;
    uint32_t ioRequestsBufferSize = 0;
    uint32_t toGPUBufferSize = 0;
    uint32_t fromGPUBufferSize = 0;

    void *mappedHeapMemory = NULL;
    void *mappedIOMemory = NULL;
    void *mappedToGPUMemory = NULL;
    void *mappedFromGPUMemory = NULL;

    uint32_t ioSize = sizeof(ioRequests);

    uint32_t localSize[3] = {1, 1, 1};

    const char *programFileName;

    uint32_t heapGlobalsOffset = 0;
    uint32_t heapGlobalsSize = 0;

    volatile bool ioRunning = true;
    volatile bool ioReset = true;
    bool runIO = true;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const char *shaderDLL;

    uint32_t *code = NULL;
    uint32_t filelength;

    uint32_t threadCount = 0;

    spirv_cross_shader_t *shader;
    const struct spirv_cross_interface *iface;

  public:

    bool verbose = false;
    bool timings = false;

    int exitCode = 0;
    uint32_t workSize[3] = {1, 1, 1};

    char *globals = NULL;
    int32_t globalsLen = 0;

    const uint32_t BufferAlign = (1 << 21);
    const uint32_t BufferAlignMinusOne = BufferAlign - 1;

    uint32_t alignBufferSize(uint32_t sz) {
        return (sz + BufferAlignMinusOne) / BufferAlign * BufferAlign;
    }

    void run(const char *fileName, int argc, char* argv[])
    {

        programFileName = fileName;

#ifdef WIN32
        _setmode(_fileno(stdout), _O_BINARY);
        _setmode(_fileno(stdin), _O_BINARY);
#endif

        timeStart();

        readShader();

        timeIval("Read shader");

        auto shaderFn = (std::string(programFileName) + ".so");
        shaderDLL = shaderFn.c_str();
        if (!fileExists(shaderDLL)) {
            if (!compileSPVToDLL(programFileName, shaderDLL)) {
                fprintf(stderr, "Failed to compile shader to dynamic library\n");
                exit(1);
            }
        }

        timeIval("Compile shader to DLL");

        // First, we get the C interface to the shader.
        // This can be loaded from a dynamic library

        void *lib = dlopen(shaderDLL, RTLD_LAZY);
        const struct spirv_cross_interface * (*spirv_cross_get_interface)(void);
        spirv_cross_get_interface = (const struct spirv_cross_interface * (*)(void)) dlsym(lib, "spirv_cross_get_interface");

        iface = (*spirv_cross_get_interface)();

        timeIval("Load shader DLL");


        threadCount = workSize[0] * workSize[1] * workSize[2] * localSize[0] * localSize[1] * localSize[2];

        int argvLen = 0;
        for (int i = 0; i < argc; i++) argvLen += strlen(argv[i]);

        ioRequestsBufferSize = ioSize;

        heapGlobalsOffset = heapBufferSize + 8;
        heapGlobalsSize = globalsLen + 8 + 8*argc + argvLen;

        heapBufferSize += heapGlobalsSize;
        toGPUBufferSize = (toGPUBufferSize < heapGlobalsSize) ? heapGlobalsSize : toGPUBufferSize;

        size_t totalIOSize = toGPUBufferSize; // + fromGPUBufferSize;
        if (verbose) fprintf(stderr, "IO buffers: %zu\n", totalIOSize);

        assert(toGPUBufferSize <= 128*(1<<20));

        ioRequestsBufferSize = alignBufferSize(ioRequestsBufferSize);
        fromGPUBufferSize = alignBufferSize(fromGPUBufferSize);
        toGPUBufferSize = alignBufferSize(toGPUBufferSize);
        heapBufferSize = alignBufferSize(heapBufferSize);

        // Create input and output buffers
        mappedHeapMemory = memalign(BufferAlign, heapBufferSize);
        if (!mappedHeapMemory) {
            fprintf(stderr, "Failed to allocate heap memory\n");
            exit(1);
        }
        mappedIOMemory = memalign(BufferAlign, ioRequestsBufferSize);
        if (!mappedIOMemory) {
            fprintf(stderr, "Failed to allocate IO requests memory\n");
            exit(1);
        }
        mappedToGPUMemory = memalign(BufferAlign, toGPUBufferSize);
        if (!mappedToGPUMemory) {
            fprintf(stderr, "Failed to allocate toGPUMemory\n");
            exit(1);
        }
        mappedFromGPUMemory = memalign(BufferAlign, fromGPUBufferSize);
        if (!mappedFromGPUMemory) {
            fprintf(stderr, "Failed to allocate fromGPUMemory\n");
            exit(1);
        }
        timeIval("Create buffers");


        // Create an instance of the shader interface.
        shader = iface->construct();

        timeIval("Create a shader instance");

        // Bind resources to the shader.
        // For resources like samplers and buffers, we provide a list of pointers,
        // since UBOs, SSBOs and samplers can be arrays, and can point to different types,
        // which is especially true for samplers.
        spirv_cross_set_resource(shader, 0, 0, &mappedHeapMemory, sizeof(mappedHeapMemory));
        spirv_cross_set_resource(shader, 0, 1, &mappedIOMemory, sizeof(mappedIOMemory));
        spirv_cross_set_resource(shader, 0, 2, &mappedToGPUMemory, sizeof(mappedToGPUMemory));
        spirv_cross_set_resource(shader, 0, 3, &mappedFromGPUMemory, sizeof(mappedFromGPUMemory));

        timeIval("Bind buffers to the shader");

        char *toGPUBuf = (char *)mappedToGPUMemory;
        ioRequests *ioReqs = (ioRequests *)mappedIOMemory;
        char *heapBuf = ((char*)mappedHeapMemory) + heapGlobalsOffset-8;
        int32_t *i32HeapBuf = (int32_t *)heapBuf;

        // Copy argv to the IO heap.
        int32_t heapEnd = heapGlobalsOffset + globalsLen;
        int32_t i32HeapEnd = heapEnd / 4;
        int32_t i32_ptr = 0;
        i32HeapBuf[i32_ptr++] = i32HeapEnd;            // start index of the argv array
        i32HeapBuf[i32_ptr++] = i32HeapEnd + argc * 2; // end index of the argv array
        if (globals != NULL) {
            memcpy(heapBuf + 8, globals, globalsLen);
        }
        int32_t heap_ptr = 8 + globalsLen + 8 * argc;
        i32_ptr = 2 + (globalsLen/4);
        int32_t heapPtr = heapEnd + 8 * argc;
        for (int i = 0; i < argc; i++) {
            int32_t len = strlen(argv[i]);
            i32HeapBuf[i32_ptr++] = heapPtr;          // start index of argv[i] on the heap
            i32HeapBuf[i32_ptr++] = heapPtr + len;    // end index of argv[i]
            memcpy(heapBuf + heap_ptr, argv[i], len); // copy argv[i] to the heap
            heap_ptr += len;
            heapPtr += len;
        }
        timeIval("Write argv and globals to GPU");

        ioReqs->ioCount = 0;
        ioReqs->programReturnValue = 0;
        ioReqs->maxIOCount = IO_REQUEST_COUNT;
        ioReqs->runCount = 0;
        ioReqs->rerunProgram = RERUN_NOW;

        std::thread ioThread;

        if (runIO) {
            ioReset = true;
            ioThread = std::thread(handleIORequests, this, verbose, ioReqs, (char*)mappedToGPUMemory, (char*)mappedFromGPUMemory, &ioRunning, &ioReset);
            while (ioReset) {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            } // Wait for ioThread initialization.
            timeIval("Start IO thread");
        }


        while (ioReqs->rerunProgram != NO_RERUN) {
            ioReqs->rerunProgram = 0;
            int64_t prevIOProgressCount = IOProgressCount;
            runProgram();
            ioReqs->runCount++;
            if (ioReqs->rerunProgram == RERUN_ON_IO) {
                while (prevIOProgressCount == IOProgressCount) {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
        }

        timeIval("Run program");

        // Call destructor.
        iface->destruct(shader);
        timeIval("Destruct shader");

        if (runIO) {
            ioRunning = false;
            ioThread.join();
            timeIval("Join IO thread");
        }

        exitCode = ioReqs->programReturnValue;
    }

    void timeStart() {
        begin = std::chrono::steady_clock::now();
    }

    void timeIval(const char *name) {
        if (verbose || timings) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            fprintf(stderr, "[%7ld us] %s\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(), name);
            begin = std::chrono::steady_clock::now();
        }
    }

    virtual void runProgram() {
        startCommandBuffer();
        waitCommandBuffer();
    }

    // This should launch a thread for the program execution.
    void startCommandBuffer() {
        // We also have to set builtins.
        // The relevant builtins will depend on the shader,
        // but for compute, there are few builtins, which are gl_NumWorkGroups and gl_WorkGroupID.
        // LocalInvocationID and GlobalInvocationID are inferred when executing the invocation.
        uvec3 num_work_groups(workSize[0], workSize[1], workSize[2]);
        uvec3 work_group_id(0, 0, 0);
        spirv_cross_set_builtin(shader, SPIRV_CROSS_BUILTIN_NUM_WORK_GROUPS, &num_work_groups, sizeof(num_work_groups));
        spirv_cross_set_builtin(shader, SPIRV_CROSS_BUILTIN_WORK_GROUP_ID, &work_group_id, sizeof(work_group_id));

        // Execute work groups.
        for (unsigned x = 0; x < num_work_groups.x; x++)
        {
            work_group_id.x = x;
            iface->invoke(shader);
        }
    }

    // This should join the program thread.
    void waitCommandBuffer() {
    }

    bool fileExists(const char *filename) {
        return std::filesystem::exists(filename);
    }

    bool compileSPVToDLL(const char *spvFilename, const char *dllFilename) {
        std::string spvFn = std::string(spvFilename);
        std::string dllFn = std::string(dllFilename);
        std::string cppFilename = spvFn + ".cpp";
        system(("spirv-cross --output " + cppFilename + " " + spvFilename + " --cpp --stage comp --vulkan-semantics").c_str());
        if (!std::filesystem::exists(cppFilename)) {
            return false;
        }
        system(("clang++ -O2 --shared -fPIC -o " + dllFn + " " + cppFilename).c_str());
        return fileExists(dllFilename);
    }

    void readFromGPUIO(int64_t offset, int64_t count) {
    }

    void cleanup() {
        free(mappedHeapMemory);
        free(mappedIOMemory);
        free(mappedToGPUMemory);
        free(mappedFromGPUMemory);
    }

    #include "io_loop.hpp"

    #include "parse_spv.hpp"
};

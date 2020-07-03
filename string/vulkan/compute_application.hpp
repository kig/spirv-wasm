#include <vulkan/vulkan.h>

#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef WIN32
#include <io.h>
#include <fcntl.h>
#endif

#define NDEBUG

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// Used for validating return values of Vulkan API calls.
#define VK_CHECK_RESULT(f)                                                                \
    {                                                                                     \
        VkResult res = (f);                                                               \
        if (res != VK_SUCCESS)                                                            \
        {                                                                                 \
            printf("Fatal : VkResult is %d in %s at line %d\n", res, __FILE__, __LINE__); \
            assert(res == VK_SUCCESS);                                                    \
        }                                                                                 \
    }

template<typename MainT, typename NewT>
inline void PnextChainPushFront(MainT* mainStruct, NewT* newStruct)
{
    newStruct->pNext = mainStruct->pNext;
    mainStruct->pNext = newStruct;
}
template<typename MainT, typename NewT>
inline void PnextChainPushBack(MainT* mainStruct, NewT* newStruct)
{
    struct VkAnyStruct
    {
        VkStructureType sType;
        void* pNext;
    };
    VkAnyStruct* lastStruct = (VkAnyStruct*)mainStruct;
    while(lastStruct->pNext != nullptr)
    {
        lastStruct = (VkAnyStruct*)lastStruct->pNext;
    }
    newStruct->pNext = nullptr;
    lastStruct->pNext = newStruct;
}

#define IO_READ 1
#define IO_WRITE 2
#define IO_CREATE 3
#define IO_DELETE 4
#define IO_TRUNCATE 5

#define IO_LISTEN 6
#define IO_ACCEPT 7
#define IO_CLOSE 8
#define IO_OPEN 9
#define IO_FSYNC 10
#define IO_SEND 11
#define IO_RECV 12
#define IO_TIMENOW 13
#define IO_TIMEOUT 14
#define IO_CD 15
#define IO_LS 16
#define IO_RMDIR 17
#define IO_CONNECT 24
#define IO_GETCWD 25
#define IO_STAT 26
#define IO_MOVE 27
#define IO_COPY 28
#define IO_COPY_RANGE 29
#define IO_MKDIR 30

#define IO_RUN_CMD 31
#define IO_EXIT 32

/*
The design for GPU IO
---

Workgroups submit tasks in sync -> readv / writev approach is beneficial for sequential reads/writes. Readv/writev are internally single-threaded, so limited by memcpy to 6-8 GB/s.

Lots of parallelism.
Ordering of writes across workgroups requires a way to sequence IOs (either reduce to order on the GPU or reassemble correct order on the CPU.)

Compression of data on the PCIe bus would help. 32 * zstd --format=lz4 --fast -T1 file -o /dev/null goes at 38 GB/s.

Need benchmark suite
---

 - Different block sizes
 - Different access patterns (sequential, random)
     - Scatter writes
     - Sequential writes
     - Gather reads
     - Sequential reads
     - Combined reads & writes
 - Different levels of parallelism
    - 1 IO per thread group
    - each thread does its own IO
    - 1 IO on ThreadID 0
    - IOs across all invocation
 - Compression
 - From hot cache on CPU
 - From cold cache
 - With GPU-side cache
 - Repeated access to same file
 - Access to multiple files

Does it help to combine reads & writes into sequential blocks on CPU-side when possible, or is it faster to do IOs ASAP?

Caching file descriptors, helps or not?



IORING_OP_NOP
// This operation does nothing at all; the benefits of doing nothing asynchronously are minimal, but sometimes a placeholder is useful.
IORING_OP_READV
IORING_OP_WRITEV
// Submit a readv() or write() operation — the core purpose for io_uring in most settings.
IORING_OP_READ_FIXED
IORING_OP_WRITE_FIXED
// These opcodes also submit I/O operations, but they use "registered" buffers that are already mapped into the kernel, reducing the amount of total overhead.
IORING_OP_FSYNC
// Issue an fsync() call — asynchronous synchronization, in other words.
IORING_OP_POLL_ADD
IORING_OP_POLL_REMOVE
// IORING_OP_POLL_ADD will perform a poll() operation on a set of file descriptors. It's a one-shot operation that must be resubmitted after it completes; it can be explicitly canceled with IORING_OP_POLL_REMOVE. Polling this way can be used to asynchronously keep an eye on a set of file descriptors. The io_uring subsystem also supports a concept of dependencies between operations; a poll could be used to hold off on issuing another operation until the underlying file descriptor is ready for it.

IORING_OP_SYNC_FILE_RANGE
// Perform a sync_file_range() call — essentially an enhancement of the existing fsync() support, though without all of the guarantees of fsync().
IORING_OP_SENDMSG
IORING_OP_RECVMSG (5.3)
// These operations support the asynchronous sending and receiving of packets over the network with sendmsg() and recvmsg().
IORING_OP_TIMEOUT
IORING_OP_TIMEOUT_REMOVE
// This operation completes after a given period of time, as measured either in seconds or number of completed io_uring operations. It is a way of forcing a waiting application to wake up even if it would otherwise continue sleeping for more completions.
IORING_OP_ACCEPT
IORING_OP_CONNECT
// Accept a connection on a socket, or initiate a connection to a remote peer.
IORING_OP_ASYNC_CANCEL
// Attempt to cancel an operation that is currently in flight. Whether this attempt will succeed depends on the type of operation and how far along it is.
IORING_OP_LINK_TIMEOUT
// Create a timeout linked to a specific operation in the ring. Should that operation still be outstanding when the timeout happens, the kernel will attempt to cancel the operation. If, instead, the operation completes first, the timeout will be canceled.

IORING_OP_FALLOCATE
// Manipulate the blocks allocated for a file using fallocate()
IORING_OP_OPENAT
IORING_OP_OPENAT2
IORING_OP_CLOSE
// Open and close files
IORING_OP_FILES_UPDATE
// Frequently used files can be registered with io_uring for faster access; this command is a way of (asynchronously) adding files to the list (or removing them from the list).
IORING_OP_STATX
// Query information about a file using statx().
IORING_OP_READ
IORING_OP_WRITE
// These are like IORING_OP_READV and IORING_OP_WRITEV, but they use the simpler interface that can only handle a single buffer.
IORING_OP_FADVISE
IORING_OP_MADVISE
// Perform the posix_fadvise() and madvise() system calls asynchronously.
IORING_OP_SEND
IORING_OP_RECV
// Send and receive network data.
IORING_OP_EPOLL_CTL
// Perform operations on epoll file-descriptor sets with epoll_ctl()

*/


#define IO_NONE 0
#define IO_START 1
#define IO_RECEIVED 2
#define IO_IN_PROGRESS 3
#define IO_COMPLETE 4
#define IO_ERROR 5
#define IO_HANDLED 255

struct ioRequest {
    int32_t ioType;
    int32_t status;
    int64_t offset;
    int64_t count;
    int32_t filename_start;
    int32_t filename_end;
    int32_t result_start;
    int32_t result_end;
};

#ifndef IO_REQUEST_COUNT
#define IO_REQUEST_COUNT 262144
#endif

struct ioRequests {
    int32_t ioCount;
    int32_t programReturnValue;
    int32_t maxIOCount;
    int32_t _pad3;
    int32_t _pad4;
    int32_t _pad5;
    int32_t _pad6;
    int32_t _pad7;
    int32_t _pad8;
    int32_t _pad9;
    ioRequest requests[IO_REQUEST_COUNT];
};


static std::mutex completed_mutex;


class ComputeApplication
{
  protected:
    VkInstance instance;

    VkDebugReportCallbackEXT debugReportCallback;

    VkPhysicalDevice physicalDevice;
    VkDevice device;

    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    VkBuffer heapBuffer;
    VkDeviceMemory heapMemory;

    VkBuffer ioRequestsBuffer;
    VkDeviceMemory ioRequestsMemory;

    VkBuffer ioHeapBuffer;
    VkDeviceMemory ioHeapMemory;

    uint32_t heapBufferSize = 0;
    uint32_t ioRequestsBufferSize = 0;
    uint32_t ioHeapBufferSize = 0;

    std::vector<const char *> enabledLayers;

    VkQueue queue;
    VkFence fence;

    void *mappedHeapMemory = NULL;
    void *mappedIOMemory = NULL;
    void *mappedIOHeapMemory = NULL;

    uint32_t queueFamilyIndex;

    uint32_t vulkanDeviceIndex = 0;

    uint32_t heapSize = 4096;
    uint32_t ioHeapSize = 4096;
    uint32_t ioSize = sizeof(ioRequests);

    uint32_t workSize[3] = {1, 1, 1};
    uint32_t localSize[3] = {1, 1, 1};

    const char *programFileName;

    uint32_t heapGlobalsOffset = 0;

    volatile bool ioRunning = true;
    volatile bool ioReset = true;
    bool runIO = true;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    bool verbose = false;
    bool timings = false;

    uint32_t *code = NULL;
    uint32_t filelength;

    uint32_t threadCount = 0;

  public:

    int exitCode = 0;

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

        threadCount = workSize[0] * workSize[1] * workSize[2] * localSize[0] * localSize[1] * localSize[2];

        heapGlobalsOffset = heapSize * threadCount;

        if (heapBufferSize == 0) heapBufferSize = heapSize * (threadCount + 1);
        if (ioHeapBufferSize == 0) ioHeapBufferSize = ioHeapSize * threadCount;
        ioRequestsBufferSize = ioSize;

        // Initialize vulkan:
        createInstance();

        findPhysicalDevice();
        timeIval("Find device");

        createDevice();


        // Create input and output buffers
        createBuffer();
        timeIval("Create buffers");

        createDescriptorSetLayout();
        timeIval("Create descriptor set layout");
        createDescriptorSet();

        createComputePipeline();

        createCommandBuffer();

        timeIval("Create command buffer");

        createFence();
        mapMemory();

        timeIval("Map memory");

        char *ioHeapBuf = (char *)mappedIOHeapMemory;
        ioRequests *ioReqs = (ioRequests *)mappedIOMemory;
        char *heapBuf = (char *)mappedHeapMemory;
        int32_t *i32HeapBuf = (int32_t *)heapBuf;

        // Copy argv to the IO heap.
        int32_t heapEnd = heapGlobalsOffset;
        int32_t i32HeapEnd = heapGlobalsOffset / 4;
        int32_t i32_ptr = i32HeapEnd;
        i32HeapBuf[i32_ptr++] = i32HeapEnd + 2;            // start index of the argv array
        i32HeapBuf[i32_ptr++] = i32HeapEnd + 2 + argc * 2; // end index of the argv array
        int32_t heapPtr = heapEnd + 4 * (2 + argc * 2);
        for (int i = 0; i < argc; i++) {
            int32_t len = strlen(argv[i]);
            i32HeapBuf[i32_ptr++] = heapPtr;          // start index of argv[i] on the heap
            i32HeapBuf[i32_ptr++] = heapPtr + len;    // end index of argv[i]
            memcpy(heapBuf + heapPtr, argv[i], len); // copy argv[i] to the heap
            heapPtr += len;
        }
        __writeHeapToGPU(heapGlobalsOffset, heapPtr - heapGlobalsOffset);

        timeIval("Write argv to GPU");

        ioReqs->ioCount = 0;
        ioReqs->programReturnValue = 0;
        ioReqs->maxIOCount = IO_REQUEST_COUNT;


        std::thread ioThread;

        if (runIO) {
            ioThread = std::thread(handleIORequests, this, verbose, ioReqs, ioHeapBuf, heapBuf, &ioRunning, &ioReset);
            timeIval("Start IO thread");
        }

        runProgram();

        timeIval("Run program");

        if (runIO) {
            ioRunning = false;
            ioThread.join();
            timeIval("Join IO thread");
        }

        exitCode = ioReqs->programReturnValue;

        unmapMemory();
        timeIval("Unmap memory");

        cleanup();
        timeIval("Clean up");
    }

    virtual void runProgram() {
        startCommandBuffer();
        waitCommandBuffer();
    }

    void timeStart() {
        begin = std::chrono::steady_clock::now();
    }

    void timeIval(const char *name) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if (verbose || timings) fprintf(stderr, "[%7ld us] %s\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(), name);
        begin = std::chrono::steady_clock::now();
    }

    void mapMemory()
    {
        vkMapMemory(device, heapMemory, 0, heapBufferSize, 0, &mappedHeapMemory);
        vkMapMemory(device, ioRequestsMemory, 0, ioRequestsBufferSize, 0, &mappedIOMemory);
        vkMapMemory(device, ioHeapMemory, 0, ioHeapBufferSize, 0, &mappedIOHeapMemory);
    }

    void unmapMemory()
    {
        vkUnmapMemory(device, heapMemory);
        vkUnmapMemory(device, ioRequestsMemory);
        vkUnmapMemory(device, ioHeapMemory);
    }

    void writeToGPU(VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size)
    {
        VkMappedMemoryRange memoryRange = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = NULL,
            .memory = memory,
            .offset = offset,
            .size = size};
        vkFlushMappedMemoryRanges(device, 1, &memoryRange);
    }

    void readFromGPU(VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size)
    {
        VkMappedMemoryRange memoryRange = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = NULL,
            .memory = memory,
            .offset = offset,
            .size = size};
        vkInvalidateMappedMemoryRanges(device, 1, &memoryRange);
    }

    void writeIOToGPU(VkDeviceSize offset, VkDeviceSize size) { writeToGPU(ioRequestsMemory, offset, size); }
    void readIOFromGPU(VkDeviceSize offset, VkDeviceSize size) { readFromGPU(ioRequestsMemory, offset, size); }

    void writeIOHeapToGPU(VkDeviceSize offset, VkDeviceSize size) { writeToGPU(ioHeapMemory, offset, size); }
    // Reading the IO heap from the CPU is extremely slow. Like, 16 MB/s slow.
    void __readIOHeapFromGPU(VkDeviceSize offset, VkDeviceSize size) { readFromGPU(ioHeapMemory, offset, size); }

    // Heap writes should only be done before starting the shader. The GPU caches memory read from the buffer.
    void __writeHeapToGPU(VkDeviceSize offset, VkDeviceSize size) { writeToGPU(heapMemory, offset, size); }
    void readHeapFromGPU(VkDeviceSize offset, VkDeviceSize size) { readFromGPU(heapMemory, offset, size); }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT flags,
        VkDebugReportObjectTypeEXT objectType,
        uint64_t object,
        size_t location,
        int32_t messageCode,
        const char *pLayerPrefix,
        const char *pMessage,
        void *pUserData)
    {

        printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

        return VK_FALSE;
    }

    void createInstance()
    {
        std::vector<const char *> enabledExtensions;

        /*
        By enabling validation layers, Vulkan will emit warnings if the API
        is used incorrectly. We shall enable the layer VK_LAYER_LUNARG_standard_validation,
        which is basically a collection of several useful validation layers.
        */
        if (enableValidationLayers)
        {
            /*
            We get all supported layers with vkEnumerateInstanceLayerProperties.
            */
            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, NULL);

            std::vector<VkLayerProperties> layerProperties(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());

            /*
            And then we simply check if VK_LAYER_LUNARG_standard_validation is among the supported layers.
            */
            bool foundLayer = false;
            for (VkLayerProperties prop : layerProperties)
            {

                if (strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0)
                {
                    foundLayer = true;
                    break;
                }
            }

            if (!foundLayer)
            {
                throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
            }
            enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation"); // Alright, we can use this layer.

            /*
            We need to enable an extension named VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
            in order to be able to print the warnings emitted by the validation layer.

            So again, we just check if the extension is among the supported extensions.
            */

            uint32_t extensionCount;

            vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, NULL);
            std::vector<VkExtensionProperties> extensionProperties(extensionCount);
            vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, extensionProperties.data());

            bool foundExtension = false;
            for (VkExtensionProperties prop : extensionProperties)
            {
                if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0)
                {
                    foundExtension = true;
                    break;
                }
            }

            if (!foundExtension)
            {
                throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
            }
            enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);

            timeIval("extensions");
        }

        VkApplicationInfo applicationInfo = {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = NULL,
            .applicationVersion = 0,
            .pEngineName = NULL,
            .engineVersion = 0,
            .apiVersion = VK_API_VERSION_1_0};

        VkInstanceCreateInfo createInfo = {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .flags = 0,
            .pApplicationInfo = &applicationInfo,

            .enabledLayerCount = static_cast<uint32_t>(enabledLayers.size()),
            .ppEnabledLayerNames = enabledLayers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size()),
            .ppEnabledExtensionNames = enabledExtensions.data()};

        VK_CHECK_RESULT(vkCreateInstance(&createInfo, NULL, &instance));
        timeIval("vkCreateInstance");

        /*
        Register a callback function for the extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME, so that warnings emitted from the validation
        layer are actually printed.
        */
        if (enableValidationLayers)
        {
            VkDebugReportCallbackCreateInfoEXT createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
            createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
            createInfo.pfnCallback = &debugReportCallbackFn;

            // We have to explicitly load this function.
            auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
            if (vkCreateDebugReportCallbackEXT == nullptr)
            {
                throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
            }

            // Create and register callback.
            VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(instance, &createInfo, NULL, &debugReportCallback));
            timeIval("vkCreateDebugReportCallbackEXT");
        }
    }

    void findPhysicalDevice()
    {
        uint32_t deviceCount;
        vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
        if (deviceCount == 0)
        {
            throw std::runtime_error("could not find a device with vulkan support");
        }

        VkPhysicalDevice devices[deviceCount];
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
        // fprintf(stderr, "%d devices, using index %d\n", deviceCount, vulkanDeviceIndex);
        physicalDevice = devices[vulkanDeviceIndex % deviceCount];
    }

    // Returns the index of a queue family that supports compute operations.
    uint32_t getComputeQueueFamilyIndex()
    {
        uint32_t queueFamilyCount;

        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

        // Retrieve all queue families.
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        // Now find a family that supports compute.
        uint32_t i = 0;
        for (; i < queueFamilies.size(); ++i)
        {
            VkQueueFamilyProperties props = queueFamilies[i];

            if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                // found a queue with compute. We're done!
                break;
            }
        }

        if (i == queueFamilies.size())
        {
            throw std::runtime_error("could not find a queue family that supports operations");
        }

        return i;
    }

    void createDevice()
    {
        queueFamilyIndex = getComputeQueueFamilyIndex(); // find queue family with compute capability.
        float queuePriorities = 1.0;                     // we only have one queue, so this is not that imporant.

        VkDeviceQueueCreateInfo queueCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamilyIndex,
            .queueCount = 1, // create one queue in this family. We don't need more.
            .pQueuePriorities = &queuePriorities};

        VkPhysicalDeviceFeatures deviceFeatures = {};

        VkDeviceCreateInfo deviceCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .enabledLayerCount = static_cast<uint32_t>(enabledLayers.size()),
            .ppEnabledLayerNames = enabledLayers.data(),
            .pQueueCreateInfos = &queueCreateInfo,
            .queueCreateInfoCount = 1,
            .pEnabledFeatures = &deviceFeatures,
            .pNext = nullptr,
        };

        VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); // create logical device.
        timeIval("vkCreateDevice");

        // Get a handle to the only member of the queue family.
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
        timeIval("vkGetDeviceQueue");
    }

    // find memory type with desired properties.
    uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memoryProperties;

        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        /*
        How does this search work?
        See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
        */
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
        {
            if ((memoryTypeBits & (1 << i)) &&
                ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
                return i;
        }
        return -1;
    }

    void createAndAllocateBuffer(VkBuffer *buffer, uint32_t bufferSize, VkDeviceMemory *bufferMemory, VkMemoryPropertyFlags flags, VkBufferUsageFlags usage)
    {
        VkBufferCreateInfo bufferCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bufferSize,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };

        VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, buffer));

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device, *buffer, &memoryRequirements);

        VkMemoryAllocateInfo allocateInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memoryRequirements.size
        };

        allocateInfo.memoryTypeIndex = findMemoryType(
            memoryRequirements.memoryTypeBits, flags);

        VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, bufferMemory));

        VK_CHECK_RESULT(vkBindBufferMemory(device, *buffer, *bufferMemory, 0));
    }


    void createBuffer()
    {
        createAndAllocateBuffer(&heapBuffer, heapBufferSize, &heapMemory, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        createAndAllocateBuffer(&ioRequestsBuffer, ioRequestsBufferSize, &ioRequestsMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
        createAndAllocateBuffer(&ioHeapBuffer, ioHeapBufferSize, &ioHeapMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    }

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[4] = {
            {.binding = 0,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 1,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
            {.binding = 2,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = descriptorSetLayoutBindings};

        // Create the descriptor set layout.
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
        //printf("createDescriptorSetLayout done\n");
    }

    void createDescriptorSet()
    {
        VkDescriptorPoolSize descriptorPoolSize = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3};

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &descriptorPoolSize};

        // create descriptor pool.
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool));
        //printf("vkCreateDescriptorPool done\n");
        timeIval("vkCreateDescriptorPool");

        /*
        With the pool allocated, we can now allocate the descriptor set.
        */
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptorSetLayout};

        // allocate descriptor set.
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));
        //printf("vkAllocateDescriptorSets done\n");
        timeIval("vkAllocateDescriptorSets");

        VkDescriptorBufferInfo heapDescriptorBufferInfo = {
            .buffer = heapBuffer,
            .offset = 0,
            .range = heapBufferSize};

        VkDescriptorBufferInfo ioDescriptorBufferInfo = {
            .buffer = ioRequestsBuffer,
            .offset = 0,
            .range = ioRequestsBufferSize};

        VkDescriptorBufferInfo ioHeapDescriptorBufferInfo = {
            .buffer = ioHeapBuffer,
            .offset = 0,
            .range = ioHeapBufferSize};

        VkWriteDescriptorSet writeDescriptorSets[4] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &heapDescriptorBufferInfo,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &ioDescriptorBufferInfo,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &ioHeapDescriptorBufferInfo,
            },
        };

        vkUpdateDescriptorSets(device, 3, writeDescriptorSets, 0, NULL);
        //printf("vkUpdateDescriptorSets done\n");
        timeIval("vkUpdateDescriptorSets");
    }

    // Read file into array of bytes, and cast to uint32_t*, then return.
    // The data has been padded, so that it fits into an array uint32_t.
    uint32_t *readFile(uint32_t &length, const char *filename)
    {
        FILE *fp = fopen(filename, "rb");
        if (fp == NULL)
        {
            printf("Could not find or open file: %s\n", filename);
        }

        // get file size.
        fseek(fp, 0, SEEK_END);
        long filesize = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        long filesizepadded = ((filesize+3) / 4) * 4;

        // read file contents.
        char *str = new char[filesizepadded];
        fread(str, filesize, sizeof(char), fp);
        fclose(fp);

        // data padding.
        for (int i = filesize; i < filesizepadded; i++)
        {
            str[i] = 0;
        }

        length = filesizepadded;
        return (uint32_t *)str;
    }

    void parseLocalSize(uint32_t *code) {
        //printf("%d\n", filelength);
        uint32_t len32 = filelength / 4;
        if (len32 <= 5) {
            fprintf(stderr, "Shader file empty: %s\n", programFileName);
            assert(len32 > 5);
        }
        uint32_t magicNumber = 0x07230203;
        assert(magicNumber == code[0]);
        for (int i = 5; i < len32; i++) {
            uint32_t op = code[i];
            uint32_t wordCount = op >> 16;
            uint32_t opCode = op & 0xffff;
            //printf("Op: %8x OpCode: %d WordCount:%d\n", op, opCode, wordCount);
            int j = i+1;
            if (opCode == 16) { // OpExecutionMode
                uint32_t entryPoint = code[j++];
                uint32_t mode = code[j++];
                //printf("EntryPoint: %d Mode: %d\n", entryPoint, mode);
                if (mode == 17) { // LocalSize
                    localSize[0] = code[j++];
                    localSize[1] = code[j++];
                    localSize[2] = code[j++];
                    //printf("LocalSize: %d %d %d\n", localSize[0], localSize[1], localSize[2]);
                }
            }
            i += wordCount > 0 ? wordCount-1 : 0;
        }
    }

    void readShader() {
        code = readFile(filelength, programFileName);
        parseLocalSize(code);
    }

    void createComputePipeline()
    {

        char* cache_data = nullptr;
        size_t cache_data_length = 0;

/*
        FILE* rf = fopen("cache", "r");
        if (rf != NULL) {
            fseek(rf, 0, SEEK_END);
            cache_data_length = ftell(rf);
            fseek(rf, 0, SEEK_SET);
//            printf("%zu\n", cache_data_length);
            cache_data = new char[cache_data_length];
            size_t datalen = fread(cache_data, 1, cache_data_length, rf);
//            printf("%zu\n", datalen);
            fclose(rf);
        }
*/
        VkPipelineCache pipeline_cache;
        VkPipelineCacheCreateInfo pipeline_cache_create_info = {
            VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            nullptr,
            0,
            cache_data_length,
            cache_data
        };
        VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipeline_cache_create_info, nullptr, &pipeline_cache));
        if (cache_data_length > 0) {
            delete[] cache_data;
        }

        timeIval("vkCreatePipelineCache");


        /*
        We create a compute pipeline here.
        */

        /*
        Create a shader module. A shader module basically just encapsulates some shader code.
        */
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pCode = code;
        createInfo.codeSize = filelength;

        VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));

        timeIval("vkCreateShaderModule");
        delete[] code;

        /*
        Now let us actually create the compute pipeline.
        A compute pipeline is very simple compared to a graphics pipeline.
        It only consists of a single stage with a compute shader.

        So first we specify the compute shader stage, and it's entry point(main).
        */
        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = computeShaderModule,
            .pName = "main"};

        /*
        The pipeline layout allows the pipeline to access descriptor sets.
        So we just specify the descriptor set layout we created earlier.
        */
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout};
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));

        timeIval("vkCreatePipelineLayout");

        VkComputePipelineCreateInfo pipelineCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = shaderStageCreateInfo,
            .layout = pipelineLayout};

        VkPipeline pl;

        /*
        Now, we finally create the compute pipeline.
        */
        VK_CHECK_RESULT(
            vkCreateComputePipelines(device, pipeline_cache, 1, &pipelineCreateInfo, NULL, &pipeline));

        timeIval("vkCreateComputePipelines");

/*
        if (cache_data_length == 0) {
            size_t data_size = 0;
            VK_CHECK_RESULT(vkGetPipelineCacheData(device, pipeline_cache, &data_size, nullptr));

            char *pipeline_cache_data = new char[data_size];
            VK_CHECK_RESULT(vkGetPipelineCacheData(device, pipeline_cache, &data_size, pipeline_cache_data));

            FILE* f = fopen("cache", "w");
            fwrite(pipeline_cache_data, 1, data_size, f);
            fclose(f);
            delete[] pipeline_cache_data;

            timeIval("vkGetPipelineCacheData");
        }
*/

    }

    void createCommandBuffer()
    {
        /*
        We are getting closer to the end. In order to send commands to the device(GPU),
        we must first record commands into a command buffer.
        To allocate a command buffer, we must first create a command pool. So let us do that.
        */
        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = 0;
        // the queue family of this command pool. All command buffers allocated from this command pool,
        // must be submitted to queues of this family ONLY.
        commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool));

        /*
        Now allocate a command buffer from the command pool.
        */
        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = commandPool; // specify the command pool to allocate from.
        // if the command buffer is primary, it can be directly submitted to queues.
        // A secondary buffer has to be called from some primary command buffer, and cannot be directly
        // submitted to a queue. To keep things simple, we use a primary command buffer.
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;                                              // allocate a single command buffer.
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer)); // allocate command buffer.

    }

    void createFence() {

        /*
          We create a fence.
        */
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = 0;
        VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, NULL, &fence));

        /*
        Now we shall finally submit the recorded command buffer to a queue.
        */

    }

    void startCommandBuffer()
    {

        //updateOutputDescriptorSet();

        /*
        Now we shall start recording commands into the newly allocated command buffer.
        */
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo)); // start recording commands.

        /*
        We need to bind a pipeline, AND a descriptor set before we dispatch.

        The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
        */
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

        /*
        Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
        The number of workgroups is specified in the arguments.
        If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
        */
        vkCmdDispatch(commandBuffer, workSize[0], workSize[1], workSize[2]);

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer)); // end recording commands.


        VkSubmitInfo submitInfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer
        };
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
    }

    void waitCommandBuffer()
    {
        /*
        The command will not have finished executing until the fence is signalled.
        So we wait here.
        We will directly after this read our buffer from the GPU,
        and we will not be sure that the command has finished executing unless we wait for the fence.
        Hence, we use a fence here.
        */
        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 1000000000000));
        VK_CHECK_RESULT(vkResetFences(device, 1, &fence));
    }

    void cleanup()
    {
        /*
        Clean up all Vulkan Resources.
        */

        if (enableValidationLayers)
        {
            // destroy callback.
            auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
            if (func == nullptr)
            {
                throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
            }
            func(instance, debugReportCallback, NULL);
        }

        vkDestroyFence(device, fence, NULL);
        timeIval("vkDestroyFence");
        vkFreeMemory(device, heapMemory, NULL);
        vkFreeMemory(device, ioRequestsMemory, NULL);
        vkFreeMemory(device, ioHeapMemory, NULL);
        timeIval("vkFreeMemory");
        vkDestroyBuffer(device, heapBuffer, NULL);
        vkDestroyBuffer(device, ioRequestsBuffer, NULL);
        vkDestroyBuffer(device, ioHeapBuffer, NULL);
        timeIval("vkDestroyBuffer");
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        timeIval("vkDestroyShaderModule");
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        timeIval("vkDestroyDescriptorPool");
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        timeIval("vkDestroyDescriptorSetLayout");
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        timeIval("vkDestroyPipeline");
        vkDestroyCommandPool(device, commandPool, NULL);
        timeIval("vkDestroyCommandPool");
        vkDestroyDevice(device, NULL);
        timeIval("vkDestroyDevice");
        vkDestroyInstance(instance, NULL);
        timeIval("vkDestroyInstance");
    }

    static FILE* openFile(char *filename, FILE* file, const char* flags) {
        if (file != NULL) return file;
        return fopen(filename, flags);
    }

    static void closeFile(FILE* file) {
        if (file != NULL) fclose(file);
    }

    static void handleIORequest(ComputeApplication *app, bool verbose, ioRequests *ioReqs, char *toGPUBuf, char *fromGPUBuf, int i, volatile bool *completed, int threadIdx) {
        while (ioReqs->requests[i].status != IO_START);
        ioRequest req = ioReqs->requests[i];
        if (verbose) fprintf(stderr, "IO req %d: t:%d s:%d off:%ld count:%ld fn_start:%d fn_end:%d res_start:%d res_end:%d\n", i, req.ioType, req.status, req.offset, req.count, req.filename_start, req.filename_end, req.result_start, req.result_end);

        // Read filename from the GPU
        char *filename = NULL;
        FILE *file = NULL;
        if (req.filename_end == 0) {
            file = stdout;
        } else {
            VkDeviceSize filenameLength = req.filename_end - req.filename_start;
            if (verbose) printf("Filename length %lu\n", filenameLength);
            filename = (char*)calloc(filenameLength + 1, 1);
            app->readHeapFromGPU(req.filename_start, filenameLength);
            memcpy(filename, fromGPUBuf + req.filename_start, filenameLength);
            filename[filenameLength] = 0;
        }

        // Process IO command
        if (req.ioType == IO_READ) {
            if (verbose) printf("Read %s\n", filename);
            auto fd = openFile(filename, file, "r");
            fseek(fd, req.offset, SEEK_SET);
            int32_t bytes = fread(toGPUBuf + req.result_start, 1, req.count, fd);
            if (file == NULL) closeFile(fd);
            app->writeIOHeapToGPU(req.result_start, bytes);
            req.result_end = req.result_start + bytes;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_WRITE) {
            auto fd = openFile(filename, file, "r+");
            app->readHeapFromGPU(req.result_start, req.count);
            if (req.offset < 0) {
                fseek(fd, -1-req.offset, SEEK_END);
            } else {
                fseek(fd, req.offset, SEEK_SET);
            }
            int32_t bytes = fwrite(fromGPUBuf + req.result_start, 1, req.count, fd);
            if (file == NULL) closeFile(fd);
            req.result_end = req.result_start + bytes;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_CREATE) {
            auto fd = openFile(filename, file, "w");
            app->readHeapFromGPU(req.result_start, req.count);
            int32_t bytes = fwrite(fromGPUBuf + req.result_start, 1, req.count, fd);
            if (file == NULL) closeFile(fd);
            req.result_end = req.result_start + bytes;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_DELETE) {
            remove(filename);
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_TRUNCATE) {
            truncate(filename, req.count);
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_RUN_CMD) {
            // Run command and copy the results to req.data
            req.status = IO_ERROR;

        } else if (req.ioType == IO_MOVE) {
            // Move file from filename to req.data
            req.status = IO_ERROR;

        } else if (req.ioType == IO_COPY) {
            // Copy file from filename to req.data
            req.status = IO_ERROR;

        } else if (req.ioType == IO_COPY_RANGE) {
            // Copy range[req.data, req.data + 8] of data from filename to (req.data + 16)
            req.status = IO_ERROR;

        } else if (req.ioType == IO_MKDIR) {
            // Make dir at filename
            req.status = IO_ERROR;

        } else if (req.ioType == IO_RMDIR) {
            // Remove dir at filename
            req.status = IO_ERROR;

        } else if (req.ioType == IO_LISTEN) {
            // Start listening for connections on TCP/UDP/Unix socket
            req.status = IO_ERROR;

        } else if (req.ioType == IO_ACCEPT) {
            // Accept connection on listening socket - async (CPU tries to fill the workgroup with sockets but might give fewer than workgroup size)
            req.status = IO_ERROR;

        } else if (req.ioType == IO_CONNECT) {
            // Connect to a remote server - async (CPU starts creating connections, GPU goes on with its merry business, GPU writes to socket are pipelined on CPU side)
            req.status = IO_ERROR;

        } else if (req.ioType == IO_SEND) {
            // Send data to a socket - async (CPU manages send queues and sends backpressure to GPU)
            // Can be run with 'udp ' uint32_t uint16_t to fire-and-forget UDP packets
            // Can be run with 'tcp ' uint32_t uint16_t to start a TCP connection, write to it, optionally read response, and close socket.
            req.status = IO_ERROR;

        } else if (req.ioType == IO_RECV) {
            // Receive data from a socket - returns data only if socket has data
            req.status = IO_ERROR;

        } else if (req.ioType == IO_EXIT) {
            req.status = IO_COMPLETE;
            exit(req.offset); // Uh..

        } else {
            req.status = IO_ERROR;

        }
        ioReqs->requests[i] = req;
        assert(ioReqs->requests[i].result_end == req.result_end);
        ioReqs->requests[i].status = req.status;
        if (verbose) printf("IO completed: %d - status %d\n", i, ioReqs->requests[i].status);

        if (threadIdx >= 0) {
            std::lock_guard<std::mutex> guard(completed_mutex);
            completed[threadIdx] = true;
        }
    }

    static void handleIORequests(ComputeApplication *app, bool verbose, ioRequests *ioReqs, char *toGPUBuf, char *fromGPUBuf, volatile bool *ioRunning, volatile bool *ioReset) {

        int threadCount = 48;
        std::thread threads[threadCount];
        int threadIdx = 0;
        volatile bool *completed = (volatile bool*)malloc(sizeof(bool) * threadCount);

        int32_t lastReqNum = 0;
        if (verbose) printf("IO Running\n");
        while (*ioRunning) {
            if (*ioReset) {
                while (threadIdx > 0) threads[--threadIdx].join();
                for (int i=0; i < threadCount; i++) completed[i] = 0;
                lastReqNum = 0;
                ioReqs->ioCount = 0;
                *ioReset = false;
            }
            int32_t reqNum = ioReqs->ioCount;
            for (int32_t i = lastReqNum; i < reqNum; i++) {
                if (verbose) printf("Got IO request %d\n", i);
                if (ioReqs->requests[i].ioType == IO_READ) {
                    int tidx = threadIdx;
                    if (tidx == threadCount) {
                        // Find completed thread.
                        for (int j = 0; j < threadCount; j++) {
                            if (completed[j]) {
                                tidx = j;
                                threads[j].join();
                                break;
                            }
                        }
                        if (tidx == threadCount) {
                            threads[0].join();
                            tidx = 0;
                        }
                    } else {
                        tidx = threadIdx++;
                    }
                    std::lock_guard<std::mutex> guard(completed_mutex);
                    completed[tidx] = false;
                    threads[tidx] = std::thread(handleIORequest, app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, tidx);
                } else {
                    handleIORequest(app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, -1);
                }
            }
            lastReqNum = reqNum;
        }
        while (threadIdx > 0) threads[--threadIdx].join();
        for (int i=0; i < threadCount; i++) completed[i] = false;
        if (verbose) printf("Exited IO thread\n");
    }

};


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

#include "../lib/io.glsl"

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
        VkResult _res_ = (f);                                                               \
        if (_res_ != VK_SUCCESS)                                                            \
        {                                                                                 \
            fprintf(stderr, "Fatal : VkResult is %d in %s at line %d\n", _res_, __FILE__, __LINE__); \
            assert(_res_ == VK_SUCCESS);                                                    \
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

    VkBuffer toGPUBuffer;
    VkDeviceMemory toGPUMemory;

    VkBuffer fromGPUBuffer;
    VkDeviceMemory fromGPUMemory;

    uint32_t heapBufferSize = 0;
    uint32_t ioRequestsBufferSize = 0;
    uint32_t toGPUBufferSize = 0;
    uint32_t fromGPUBufferSize = 0;

    std::vector<const char *> enabledLayers;

    VkQueue queue;
    VkFence fence;

    VkCommandPool copyCommandPool;
    VkCommandBuffer copyCommandBuffer;
    VkFence copyFence;
    VkQueue copyQueue;

    void *mappedHeapMemory = NULL;
    void *mappedIOMemory = NULL;
    void *mappedToGPUMemory = NULL;
    void *mappedFromGPUMemory = NULL;

    uint32_t queueFamilyIndex;

    uint32_t vulkanDeviceIndex = 0;

    uint32_t ioSize = sizeof(ioRequests);

    uint32_t localSize[3] = {1, 1, 1};

    const char *programFileName;

    uint32_t heapGlobalsOffset = 0;
    uint32_t heapGlobalsSize = 0;

    volatile bool ioRunning = true;
    volatile bool ioReset = true;
    bool runIO = true;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    uint32_t *code = NULL;
    uint32_t filelength;

    uint32_t threadCount = 0;

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

        // Initialize vulkan
        createInstance();

        findPhysicalDevice();
        timeIval("Find device");

        createDevice();

        ioRequestsBufferSize = alignBufferSize(ioRequestsBufferSize);
        fromGPUBufferSize = alignBufferSize(fromGPUBufferSize);
        toGPUBufferSize = alignBufferSize(toGPUBufferSize);
        heapBufferSize = alignBufferSize(heapBufferSize);

        // Create input and output buffers
        createBuffer();
        timeIval("Create buffers");

        createDescriptorSetLayout();
        createDescriptorSet();

        createComputePipeline();

        createCommandBuffer();
        timeIval("Create command buffer");

        createFence();
        timeIval("Create fence");

        initCommandBuffer(queueFamilyIndex, &copyCommandPool, &copyCommandBuffer);
        timeIval("Create copy command buffer");
        initFence(&copyFence);
        timeIval("Create copy fence");

        mapMemory();

        timeIval("Map memory");

        char *toGPUBuf = (char *)mappedToGPUMemory;
        ioRequests *ioReqs = (ioRequests *)mappedIOMemory;
        char *heapBuf = toGPUBuf;
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
        bufferCopy(heapBuffer, heapGlobalsOffset-8, toGPUBuffer, 0, heapGlobalsSize);
//            VkBufferCopy copyRegion{ .srcOffset = 0, .dstOffset = heapGlobalsOffset-8, .size = heapGlobalsSize };
//            vkCmdCopyBuffer(commandBuffer, toGPUBuffer, heapBuffer, 1, &copyRegion);
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
            // If rerunProgram is set at the end of the program, this may not be set yet on the CPU side.
            // How to fix? Invalidate first req.
            readFromGPU(ioRequestsMemory, 0, 128);
            if (ioReqs->rerunProgram == RERUN_ON_IO) {
                while (prevIOProgressCount == IOProgressCount) {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
        }

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
        if (verbose || timings) {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            fprintf(stderr, "[%7ld us] %s\n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(), name);
            begin = std::chrono::steady_clock::now();
        }
    }

    void mapMemory()
    {
        vkMapMemory(device, heapMemory, 0, heapBufferSize, 0, &mappedHeapMemory);
        vkMapMemory(device, fromGPUMemory, 0, fromGPUBufferSize, 0, &mappedFromGPUMemory);
        vkMapMemory(device, toGPUMemory, 0, toGPUBufferSize, 0, &mappedToGPUMemory);
        vkMapMemory(device, ioRequestsMemory, 0, ioRequestsBufferSize, 0, &mappedIOMemory);
    }

    void unmapMemory()
    {
        vkUnmapMemory(device, heapMemory);
        vkUnmapMemory(device, ioRequestsMemory);
        vkUnmapMemory(device, toGPUMemory);
        vkUnmapMemory(device, fromGPUMemory);
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

    void readFromGPUIO(VkDeviceSize offset, VkDeviceSize size)
    {
        readFromGPU(fromGPUMemory, offset, size);
    }

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
            enabledExtensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
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
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);

        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        uint32_t i = 0;
        for (; i < queueFamilies.size(); ++i) {
            VkQueueFamilyProperties props = queueFamilies[i];
            if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                break;
            }
        }

        if (i == queueFamilies.size()) {
            throw std::runtime_error("Could not find a queue family that supports compute.");
        }

        return i;
    }

    void createDevice()
    {
        queueFamilyIndex = getComputeQueueFamilyIndex();
        float queuePriorities = 1.0;

        VkDeviceQueueCreateInfo queueCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamilyIndex,
            .queueCount = 2, // One queue for compute, one for buffer copies.
            .pQueuePriorities = &queuePriorities};

        VkPhysicalDeviceFeatures features1 = {};

        VkPhysicalDeviceFeatures2 features = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = NULL, .features = features1, };
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features);

        VkDeviceCreateInfo deviceCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &features,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledLayerCount = static_cast<uint32_t>(enabledLayers.size()),
            .ppEnabledLayerNames = enabledLayers.data(),
            .pEnabledFeatures = NULL,
        };

        VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device));
        timeIval("vkCreateDevice");

        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
        vkGetDeviceQueue(device, queueFamilyIndex, 1, &copyQueue);
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
        createAndAllocateBuffer(&toGPUBuffer, toGPUBufferSize, &toGPUMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        createAndAllocateBuffer(&ioRequestsBuffer, ioRequestsBufferSize, &ioRequestsMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
        createAndAllocateBuffer(&fromGPUBuffer, fromGPUBufferSize, &fromGPUMemory, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        createAndAllocateBuffer(&heapBuffer, heapBufferSize, &heapMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
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
            {.binding = 3,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 4,
            .pBindings = descriptorSetLayoutBindings};

        // Create the descriptor set layout.
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
        timeIval("vkCreateDescriptorSetLayout");
    }

    void createDescriptorSet()
    {
        VkDescriptorPoolSize descriptorPoolSize = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 4};

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &descriptorPoolSize};

        // create descriptor pool.
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool));
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
        timeIval("vkAllocateDescriptorSets");

        VkDescriptorBufferInfo heapDescriptorBufferInfo = {
            .buffer = heapBuffer,
            .offset = 0,
            .range = heapBufferSize};

        VkDescriptorBufferInfo ioDescriptorBufferInfo = {
            .buffer = ioRequestsBuffer,
            .offset = 0,
            .range = ioRequestsBufferSize};

        VkDescriptorBufferInfo toGPUDescriptorBufferInfo = {
            .buffer = toGPUBuffer,
            .offset = 0,
            .range = toGPUBufferSize};

        VkDescriptorBufferInfo fromGPUDescriptorBufferInfo = {
            .buffer = fromGPUBuffer,
            .offset = 0,
            .range = fromGPUBufferSize};

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
                .pBufferInfo = &toGPUDescriptorBufferInfo,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet,
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &fromGPUDescriptorBufferInfo,
            },
        };

        vkUpdateDescriptorSets(device, 4, writeDescriptorSets, 0, NULL);
        timeIval("vkUpdateDescriptorSets");
    }

    void createComputePipeline()
    {

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
            vkCreateComputePipelines(device, NULL, 1, &pipelineCreateInfo, NULL, &pipeline));

        timeIval("vkCreateComputePipelines");
    }

    void initCommandBuffer(uint32_t queueFamilyIndex, VkCommandPool *commandPool, VkCommandBuffer *commandBuffer)
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = 0, .queueFamilyIndex = queueFamilyIndex};
        VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, commandPool));

        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = *commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1, };
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffer));
    }

    void initFence(VkFence *fence) {
        VkFenceCreateInfo fenceCreateInfo = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = 0};
        VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, NULL, fence));
    }

    void createCommandBuffer() { initCommandBuffer(queueFamilyIndex, &commandPool, &commandBuffer); }
    void createFence() { initFence(&fence); }
    void startCommandBuffer()
    {
        VkCommandBufferBeginInfo beginInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT};
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

            /*
            We need to bind a pipeline, AND a descriptor set before we dispatch.
            The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
            */
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

            vkCmdDispatch(commandBuffer, workSize[0], workSize[1], workSize[2]);

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer)); // end recording commands.

        VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &commandBuffer };
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
    }

    void waitForFence(VkFence *fence) {
        while(true) {
            VkResult res = vkWaitForFences(device, 1, fence, VK_TRUE, 1000000000);
            if (res == VK_SUCCESS) break;
            if (res != VK_TIMEOUT) {
                VK_CHECK_RESULT(res);
                break;
            }
        }
        VK_CHECK_RESULT(vkResetFences(device, 1, fence));
    }

    void waitCommandBuffer() { waitForFence(&fence); }


    void bufferCopy(VkBuffer dst, size_t dstOff, VkBuffer src, size_t srcOff, size_t byteCount) {
        VkCommandBufferBeginInfo beginInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT};
        VK_CHECK_RESULT(vkBeginCommandBuffer(copyCommandBuffer, &beginInfo));

            VkBufferCopy copyRegion{ .srcOffset = srcOff, .dstOffset = dstOff, .size = byteCount };
            vkCmdCopyBuffer(copyCommandBuffer, src, dst, 1, &copyRegion);

        VK_CHECK_RESULT(vkEndCommandBuffer(copyCommandBuffer));

        VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &copyCommandBuffer };
        VK_CHECK_RESULT(vkQueueSubmit(copyQueue, 1, &submitInfo, copyFence));

        waitForFence(&copyFence);
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
        vkFreeMemory(device, toGPUMemory, NULL);
        vkFreeMemory(device, fromGPUMemory, NULL);
        timeIval("vkFreeMemory");
        vkDestroyBuffer(device, heapBuffer, NULL);
        vkDestroyBuffer(device, ioRequestsBuffer, NULL);
        vkDestroyBuffer(device, toGPUBuffer, NULL);
        vkDestroyBuffer(device, fromGPUBuffer, NULL);
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
        vkDestroyCommandPool(device, copyCommandPool, NULL);
        timeIval("vkDestroyCommandPool");
        vkDestroyDevice(device, NULL);
        timeIval("vkDestroyDevice");
        vkDestroyInstance(instance, NULL);
        timeIval("vkDestroyInstance");
    }

    #include "io_loop.hpp"

    #include "parse_spv.hpp"
};


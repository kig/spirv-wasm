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
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <zstd.h>
#include <lz4.h>
#include <lz4frame.h>
#include <dlfcn.h>

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
    int32_t _pad3;
    int32_t _pad4;
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
        timeIval("Write argv and globals to GPU");

        ioReqs->ioCount = 0;
        ioReqs->programReturnValue = 0;
        ioReqs->maxIOCount = IO_REQUEST_COUNT;


        std::thread ioThread;

        if (runIO) {
            ioThread = std::thread(handleIORequests, this, verbose, ioReqs, (char*)mappedToGPUMemory, (char*)mappedFromGPUMemory, &ioRunning, &ioReset);
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

        VkPhysicalDeviceFeatures deviceFeatures = {};

        VkDeviceCreateInfo deviceCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledLayerCount = static_cast<uint32_t>(enabledLayers.size()),
            .ppEnabledLayerNames = enabledLayers.data(),
            .pEnabledFeatures = &deviceFeatures,
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

#define TAG(v) (((v>>24) & 0xff) | ((v>>8) & 0xff00) | (((uint32_t)v<<8) & 0xff0000) | (((uint32_t)v << 24) & 0xff000000))

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
            #ifndef NDEBUG
            fprintf(stderr, "Op: %8x OpCode: %d WordCount:%d\n", op, opCode, wordCount);
            #endif
            int j = i+1;
            if (opCode == 16) { // OpExecutionMode
                uint32_t entryPoint = code[j++];
                uint32_t mode = code[j++];
                if (verbose) fprintf(stderr, "EntryPoint: %d Mode: %d\n", entryPoint, mode);
                if (mode == 17) { // LocalSize
                    localSize[0] = code[j++];
                    localSize[1] = code[j++];
                    localSize[2] = code[j++];
                    if (verbose) fprintf(stderr, "LocalSize: %d %d %d\n", localSize[0], localSize[1], localSize[2]);
                }
            }
            if (opCode == 4) { // OpSourceExtension
                int j = i + 1;
                uint32_t tag = code[j++];
                if (verbose) fprintf(stderr, "OpSourceExtension tag %.4s %d\n", (char*)(&tag), code[j]);
                if (tag == TAG('glo=')) {
                    const uint32_t len = 4 * (wordCount - 2);
                    globalsLen += len;
                    if (globals != NULL) globals = (char*)realloc(globals, globalsLen+1);
                    else globals = (char *)malloc(globalsLen+1);
                    memcpy(globals + (globalsLen-len), code + j, len);
                    globals[globalsLen] = 0;
                    if (verbose) fprintf(stderr, "globals[%d]: %s\n", globalsLen, globals);
                }
                else if (tag == TAG('tgc=')) workSize[0] = code[j++];
                else if (tag == TAG('ths=')) heapBufferSize = code[j++];
                else if (tag == TAG('tti=')) fromGPUBufferSize = code[j++];
                else if (tag == TAG('tfi=')) toGPUBufferSize = code[j++];
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

            VkBufferCopy copyRegion{ .srcOffset = 0, .dstOffset = heapGlobalsOffset-8, .size = heapGlobalsSize };
            vkCmdCopyBuffer(commandBuffer, toGPUBuffer, heapBuffer, 1, &copyRegion);

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

    static FILE* openFile(char *filename, FILE* file, const char* flags) {
        if (file != NULL) return file;
        return fopen(filename, flags);
    }

    static void closeFile(FILE* file) {
        if (file != NULL) fclose(file);
    }

    static void handleIORequest(ComputeApplication *app, bool verbose, ioRequests *ioReqs, char *toGPUBuf, char *fromGPUBuf, int i, volatile bool *completed, int threadIdx, std::map<int, FILE*> *fileCache) {
        volatile ioRequest *volatileReqs = ((volatile ioRequest*)(ioReqs->requests));
        while (volatileReqs[i].status != IO_START) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        volatileReqs[i].status = IO_IN_PROGRESS;
        ioRequest req = ioReqs->requests[i];
        if (verbose) fprintf(stderr, "IO req %d: t:%d s:%d off:%ld count:%ld fn_start:%d fn_end:%d res_start:%d res_end:%d\n", i, req.ioType, req.status, req.offset, req.count, req.filename_start, req.filename_end, req.result_start, req.result_end);

        // Read filename from the GPU
        char *filename = NULL;
        FILE *file = NULL;
        if (req.filename_end == 0) {
            if (req.filename_start == 0) {
                file = stdin;
            } else if (req.filename_start == 1) {
                file = stdout;
            } else if (req.filename_start == 2) {
                file = stderr;
            } else {
                std::lock_guard<std::mutex> guard(fileCache_mutex);
                file = fileCache->at(req.filename_start);
            }
        } else {
            VkDeviceSize filenameLength = req.filename_end - req.filename_start;
            if (verbose) fprintf(stderr, "Filename length %lu\n", filenameLength);
            filename = (char*)calloc(filenameLength + 1, 1);
            app->readFromGPUIO(req.filename_start, filenameLength);
            memcpy(filename, fromGPUBuf + req.filename_start, filenameLength);
            filename[filenameLength] = 0;
        }

        // Process IO command
        if (req.ioType == IO_READ) {
            if (verbose) fprintf(stderr, "Read %s\n", filename);
            auto fd = openFile(filename, file, "rb");
            fseek(fd, req.offset, SEEK_SET);

            int32_t bytes = 0;
            int64_t totalRead = 0;

            const int32_t compressionType = req.compression & 0xff000000;
            const int32_t compressionData = req.compression & 0x00ffffff;
            const int32_t blockSize = compressionData & 0x000fffff;
            int32_t compressionSpeed = ((compressionData & 0x00f00000) >> 20) - 1;
            bool autoCompress = compressionSpeed == -1;
            if (compressionSpeed == -1) compressionSpeed = 7;

            if (verbose) fprintf(stderr, "Compression: %08x, type: %d, speed: %d, blockSize: %d\n", req.compression, compressionType, compressionSpeed, blockSize);

            bool doUncompressedRead = true;

            if (compressionType != 0) {

                doUncompressedRead = false;

                if (compressionType == IO_COMPRESS_ZSTD) {
                    size_t const buffInSize = ZSTD_CStreamInSize();
                    void*  const buffIn  = malloc(buffInSize);
                    size_t const buffOutSize = ZSTD_CStreamOutSize();
                    void*  const buffOut = malloc(buffOutSize);

                    int minLevel = ZSTD_minCLevel();
                    int maxLevel = ZSTD_maxCLevel();
                    float speedRatio = sqrt(compressionSpeed/9.0);
                    int level = round(maxLevel * (1-speedRatio) + minLevel * speedRatio);

                    ZSTD_CCtx* const cctx = ZSTD_createCCtx();
                    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
                    ZSTD_CCtx_setParameter(cctx, ZSTD_c_strategy, ZSTD_fast);

                    size_t readCount = req.count;
                    size_t toRead = readCount < buffInSize ? readCount : buffInSize;
                    for (;;) {
                        size_t read = fread(buffIn, 1, readCount < toRead ? readCount : toRead, fd);
                        readCount -= read;
                        totalRead += read;
                        bool lastChunk = (read < toRead || readCount == 0);
                        ZSTD_EndDirective const mode = lastChunk ? ZSTD_e_end : ZSTD_e_continue;
                        ZSTD_inBuffer input = { buffIn, read, 0 };
                        bool finished = false;
                        do {
                            ZSTD_outBuffer output = { buffOut, buffOutSize, 0 };
                            size_t const remaining = ZSTD_compressStream2(cctx, &output, &input, mode);
                            memcpy(toGPUBuf + req.result_start + bytes, buffOut, output.pos);
                            bytes += output.pos;
                            finished = lastChunk ? (remaining == 0) : (input.pos == input.size);
                            volatileReqs[i].progress = bytes;
                        } while (!finished);
                        if (lastChunk) break;
                    }
                    if (verbose) fprintf(stderr, "Wrote %ld => %d bytes\n", req.count, bytes);

                    ZSTD_freeCCtx(cctx);
                    free(buffIn);
                    free(buffOut);
                    req.status = IO_COMPLETE;

                } else if (compressionType == IO_COMPRESS_LZ4_BLOCK || compressionType == IO_COMPRESS_LZ4_BLOCK_STREAM) {
                    // Use LZ4 block compression

                    LZ4_stream_t lz4Stream_body;
                    LZ4_stream_t* lz4Stream = &lz4Stream_body;

                    const int64_t BLOCK_BYTES = 8192;

                    char inpBuf[2][BLOCK_BYTES + 8];
                    char cmpBuf[LZ4_COMPRESSBOUND(BLOCK_BYTES) + 8];
                    int64_t inpBufIndex = 0;

                    LZ4_initStream(lz4Stream, sizeof (*lz4Stream));

                    int64_t readCount = req.count;

                    if (compressionType == IO_COMPRESS_LZ4_BLOCK_STREAM) { // LZ4_BLOCK_STREAM

                        int32_t blockOff = 0;
                        int32_t blockBytes = 0;

                        const int32_t blockCount = 128;
                        bytes = 128 * 4;

                        for (int i = 0; i < blockCount; i++) {
                            *(int32_t*)(toGPUBuf + req.result_start + i*4) = 0;
                        }

                        if (verbose) fprintf(stderr, "Compression frame size: %d, internal block size: %ld\n", blockSize, BLOCK_BYTES);
                        if (verbose) fprintf(stderr, "Compression speed: %d\n", compressionSpeed);

                        while (true) {
                            char* const inpPtr = inpBuf[inpBufIndex];
                            const int64_t inpBytes = (int) fread(inpPtr, 1, std::min(BLOCK_BYTES, (readCount - totalRead)), fd);
                            totalRead += inpBytes;
                            if (0 == inpBytes) break;
                            int64_t cmpBytes = LZ4_compress_fast_continue(lz4Stream, inpPtr, cmpBuf, inpBytes, sizeof(cmpBuf), compressionSpeed);
                            if (cmpBytes <= 0) break;

                            if (totalRead == inpBytes && autoCompress) {
                                float compressionRatio = (float)cmpBytes / inpBytes;
                                if (blockOff == 0) {
                                    if (compressionRatio > 0.7) {
                                        // Do uncompressed read
                                        memcpy(toGPUBuf + req.result_start, inpPtr, inpBytes);
                                        volatileReqs[i].compression = 0;
                                        volatileReqs[i].progress = inpBytes;
                                        bytes = inpBytes;
                                        doUncompressedRead = true;
                                        break;
                                    } else if (compressionRatio < 0.3) {
                                        // Use stronger compression
                                        compressionSpeed = 3;
                                        // Recompress first block?
                                        // LZ4_resetStream_fast(lz4Stream);
                                        // cmpBytes = LZ4_compress_fast_continue(lz4Stream, inpPtr, cmpBuf, inpBytes, sizeof(cmpBuf), compressionSpeed);
                                    } else {
                                        // Use faster compression
                                        compressionSpeed = 7;
                                    }
                                }
                            }

                            memcpy(toGPUBuf + req.result_start + bytes, cmpBuf, cmpBytes);
                            bytes += cmpBytes;
                            blockBytes += cmpBytes;
                            if ((totalRead & (blockSize-1)) == 0) {
                                *(uint32_t*)(toGPUBuf + req.result_start + blockOff) = blockBytes;
                                if (verbose) fprintf(stderr, "%d\n", blockBytes);
                                blockBytes = 0;
                                blockOff += 4;
                                volatileReqs[i].progress = bytes;
                                LZ4_resetStream_fast(lz4Stream);
                            }
                            inpBufIndex = (inpBufIndex + 1) % 2;
                        }
                        if (!doUncompressedRead) {
                             *(int32_t*)(toGPUBuf + req.result_start + blockOff) = blockBytes;
                            if (verbose && blockBytes > 0) fprintf(stderr, "[%d] = %d\n", req.result_start + blockOff, blockBytes);
                            if (verbose && totalRead > 0) fprintf(stderr, "Compression ratio: %.4f\n", (float)bytes/totalRead);
                            req.status = IO_COMPLETE;
                        }


                    } else { // LZ4_BlOCK
                        while (true) {
                            char* const inpPtr = inpBuf[inpBufIndex];
                            const int inpBytes = (int) fread(inpPtr, 1, std::min(BLOCK_BYTES, (readCount - totalRead)), fd);
                            totalRead += inpBytes;
                            if (0 == inpBytes) break;
                            const int cmpBytes = LZ4_compress_fast_continue(lz4Stream, inpPtr, cmpBuf, inpBytes, sizeof(cmpBuf), compressionSpeed);
                            if (cmpBytes <= 0) break;
                            memcpy(toGPUBuf + req.result_start + bytes, cmpBuf, cmpBytes);
                            bytes += cmpBytes;
                            volatileReqs[i].progress = bytes;
                            inpBufIndex = (inpBufIndex + 1) % 2;
                        }
                        if (verbose && totalRead > 0) fprintf(stderr, "Compression ratio: %.4f\n", (float)bytes/totalRead);
                        req.status = IO_COMPLETE;
                    }


                } else if (compressionType == IO_COMPRESS_LZ4) { // LZ4_FRAME_STREAM

                    if (verbose) fprintf(stderr, "IO_COMPRESS_LZ4\n");

                    const size_t inChunkSize = 65536;

                    const LZ4F_preferences_t kPrefs = {
                        { LZ4F_max256KB, LZ4F_blockLinked, LZ4F_noContentChecksum, LZ4F_frame,
                          0 /* unknown content size */, 0 /* no dictID */ , LZ4F_noBlockChecksum },
                         -3,   /* compression level; 0 == default */
                          0,   /* autoflush */
                          1,   /* favor decompression speed */
                          { 0, 0, 0 },  /* reserved, must be set to 0 */
                    };

                    LZ4F_compressionContext_t ctx;
                    size_t const ctxCreation = LZ4F_createCompressionContext(&ctx, LZ4F_VERSION);
                    void* const inBuff = malloc(inChunkSize);
                    size_t const outCapacity = LZ4F_compressBound(inChunkSize, &kPrefs);
                    void* const outBuff = malloc(outCapacity);

                    if (!LZ4F_isError(ctxCreation) && inBuff && outBuff) {
                        uint64_t count_in = 0, count_out = 0;

                        assert(ctx != NULL);
                        assert(outCapacity >= LZ4F_HEADER_SIZE_MAX);
                        assert(outCapacity >= LZ4F_compressBound(inChunkSize, &kPrefs));

                        /* write frame header */
                        {   size_t const headerSize = LZ4F_compressBegin(ctx, outBuff, outCapacity, &kPrefs);
                            if (LZ4F_isError(headerSize)) {
                                fprintf(stderr, "Error: LZ4F Failed to start compression: error %u \n", (unsigned)headerSize);
                                exit(1);
                            }
                            memcpy(toGPUBuf + req.result_start + count_out, outBuff, headerSize);
                            count_out = headerSize;
                            volatileReqs[i].progress = count_out;
                            if (verbose) fprintf(stderr, "Buffer size is %u bytes, header size %u bytes \n",
                                    (unsigned)outCapacity, (unsigned)headerSize);
                        }

                        /* stream file */
                        for (;;) {
                            size_t const readSize = fread(inBuff, 1, std::min(inChunkSize, (req.count - count_in)), fd);
                            if (readSize == 0) break; /* nothing left to read from input file */
                            count_in += readSize;

                            size_t const compressedSize = LZ4F_compressUpdate(ctx,
                                                                    outBuff, outCapacity,
                                                                    inBuff, readSize,
                                                                    NULL);
                            if (LZ4F_isError(compressedSize)) {
                                fprintf(stderr, "Error: LZ4F Compression failed: error %u \n", (unsigned)compressedSize);
                                exit(1);
                            }

                            if (verbose) fprintf(stderr, "Writing %u bytes\n", (unsigned)compressedSize);
                            memcpy(toGPUBuf + req.result_start + count_out, outBuff, compressedSize);
                            count_out += compressedSize;
                            volatileReqs[i].progress = count_out;
                        }

                        /* flush whatever remains within internal buffers */
                        {   size_t const compressedSize = LZ4F_compressEnd(ctx,
                                                                    outBuff, outCapacity,
                                                                    NULL);
                            if (LZ4F_isError(compressedSize)) {
                                fprintf(stderr, "Error: LZ4F Failed to end compression: error %u\n", (unsigned)compressedSize);
                                exit(1);
                            }

                            if (verbose) fprintf(stderr, "Writing %u bytes \n", (unsigned)compressedSize);
                            memcpy(toGPUBuf + req.result_start + count_out, outBuff, compressedSize);
                            count_out += compressedSize;
                            volatileReqs[i].progress = count_out;
                        }

                        bytes = count_out;
                        totalRead = count_in;
                        req.status = IO_COMPLETE;

                    } else {
                        req.status = IO_ERROR;
                        fprintf(stderr, "Error: LZ4F resource allocation failed\n");
                        exit(1);
                    }

                    free(outBuff);
                    free(inBuff);
                    LZ4F_freeCompressionContext(ctx);   /* supports free on NULL */

                } else { // Unknown compression type
                    req.status = IO_ERROR;
                }

            }

            if (doUncompressedRead) { // Uncompressed read.

                int64_t count = req.count;
                while (bytes < req.count) {
                    int32_t readBytes = fread(toGPUBuf + req.result_start + bytes, 1, std::min(req.count-bytes, 65536L), fd);
                    bytes += readBytes;
                    volatileReqs[i].progress = bytes;
                    if (readBytes == 0) break;
                }
                totalRead = (int64_t)bytes;
                if (verbose) fprintf(stderr, "Read %d = %ld bytes to GPU\n", bytes, totalRead);
                req.status = IO_COMPLETE;
            }

            if (file == NULL) { // Request reading file to page cache
                fseek(fd, 0, SEEK_END);
                readahead(fileno(fd), 0, ftell(fd));
            }
            if (file == NULL) closeFile(fd);

            volatileReqs[i].count = totalRead;
            volatileReqs[i].result_end = req.result_start + bytes;
            if (verbose) fprintf(stderr, "Sent count %ld = %ld to GPU\n", totalRead, volatileReqs[i].count);

        } else if (req.ioType == IO_WRITE) {
            struct stat st;
            int mode = 0;
            if (0 != stat(filename, &st)) {
                mode = 1;
            }
            errno = 0;
            if (verbose) fprintf(stderr, "openFile %s with mode %d\n", filename, mode);
            auto fd = openFile(filename, file, mode == 0 ? "rb+" : "w");
            app->readFromGPUIO(req.result_start, req.count);
            if (req.offset < 0) {
                fseek(fd, -1-req.offset, SEEK_END);
            } else {
                fseek(fd, req.offset, SEEK_SET);
            }
            int32_t bytes = fwrite(fromGPUBuf + req.result_start, 1, req.count, fd);
            if (file == NULL) closeFile(fd);
            volatileReqs[i].result_end = req.result_start + bytes;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_CREATE) {
            // Create new file
            auto fd = openFile(filename, file, "wb");
            app->readFromGPUIO(req.result_start, req.count);
            int32_t bytes = fwrite(fromGPUBuf + req.result_start, 1, req.count, fd);
            if (file == NULL) closeFile(fd);
            volatileReqs[i].result_end = req.result_start + bytes;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_DELETE) {
            // Delete file
            remove(filename);
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_TRUNCATE) {
            // Truncate file
            truncate(filename, req.count);
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_NOP) {
            // Do nothing
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_PINGPONG) {
            // Send the received data back
            app->readFromGPUIO(req.result_start, req.count);
            memcpy(toGPUBuf + req.result_start, fromGPUBuf + req.offset, req.count);
            volatileReqs[i].result_end = req.result_start + req.count;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_RUN_CMD) {
            // Run command and copy the results to req.data
            int result = system(filename);
            volatileReqs[i].result_start = result;
            volatileReqs[i].result_end = 0;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_MOVE) {
            // Move file from filename to req.data
            char *dst = (char*)calloc(req.count+1, 1);
            app->readFromGPUIO(req.result_start, req.count);
            memcpy(dst, fromGPUBuf + req.result_start, req.count);
            dst[req.count] = 0;
            int result = rename(filename, dst);
            volatileReqs[i].result_start = result;
            volatileReqs[i].result_end = 0;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_COPY) {
            // Copy file from filename to req.data
            char *dst = (char*)calloc(req.count+1, 1);
            app->readFromGPUIO(req.result_start, req.count);
            memcpy(dst, fromGPUBuf + req.result_start, req.count);
            dst[req.count] = 0;
            std::error_code ec;
            std::filesystem::copy(filename, dst, ec);
            volatileReqs[i].result_start = ec.value();
            volatileReqs[i].result_end = 0;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_COPY_RANGE) {
            // Copy range[req.data, req.data + 8] of data from filename to (req.data + 16)
            req.status = IO_ERROR;

        } else if (req.ioType == IO_CD) {
            // Change directory
            int result = chdir(filename);
            volatileReqs[i].result_start = result;
            volatileReqs[i].result_end = 0;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_MKDIR) {
            // Make dir at filename
            int result = mkdir(filename, 0755);
            volatileReqs[i].result_start = result;
            volatileReqs[i].result_end = 0;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_RMDIR) {
            // Remove dir at filename
            int result = rmdir(filename);
            volatileReqs[i].result_start = result;
            volatileReqs[i].result_end = 0;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_GETCWD) {
            // Get current working directory
            char *s = getcwd(NULL, req.count);
            int32_t len = strlen(s);
            printf("getcwd: %s\n", s);
            memcpy(toGPUBuf + req.result_start, s, len);
            free(s);
            volatileReqs[i].result_end = req.result_start + len;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_OPEN) {
            // Open file
            FILE* fd = fopen(filename, req.offset == 0 ? "rb" : (req.offset == 1 ? "rb+" : (req.offset == 2 ? "rw" : "ra")));
            *(int32_t*)(toGPUBuf + req.result_start) = fileno(fd);
            volatileReqs[i].result_end = req.result_start + 4;
            std::lock_guard<std::mutex> guard(fileCache_mutex);
            fileCache->emplace(fileno(fd), fd);
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_CLOSE) {
            // Close open file
            if (file) {
                fclose(file);
                std::lock_guard<std::mutex> guard(fileCache_mutex);
                fileCache->erase(req.filename_start);
            }
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_LS) {
            // List files with filename
            uint32_t count = 0, offset = 0, entryCount = 0;
            for (const auto & entry : std::filesystem::directory_iterator(filename)) {
                entryCount++;
                if (count >= req.count) {
                    continue;
                }
                if (offset >= req.offset) {
                    auto path = entry.path();
                    const char* s = path.c_str();
                    int32_t len = strlen(s);
                    if (count + 4 + len >= req.count) break;
                    *(int32_t*)(toGPUBuf + req.result_start + count) = len;
                    count += 4;
                    memcpy(toGPUBuf + req.result_start + count, s, len);
                    count += len;
                }
                offset++;
            }
            volatileReqs[i].offset = offset;
            volatileReqs[i].count = entryCount;
            volatileReqs[i].result_end = req.result_start + count;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_STAT) {
            // Stat a path and return file info
            struct stat st;
            int ok;
            if (req.offset == 1) {
                ok = lstat(filename, &st);
            } else {
                ok = stat(filename, &st);
            }
            if (verbose) fprintf(stderr, "Got %d from stat, req.count = %ld\n", ok, req.count);
            if (ok == 0 && req.count >= 104) {
                uint64_t *u64 = (uint64_t*)(toGPUBuf + req.result_start);
                int i = 0;
                u64[i++] = st.st_atim.tv_sec;
                u64[i++] = st.st_atim.tv_nsec;
                u64[i++] = st.st_mtim.tv_sec;
                u64[i++] = st.st_mtim.tv_nsec;
                u64[i++] = st.st_ctim.tv_sec;
                u64[i++] = st.st_ctim.tv_nsec;

                u64[i++] = st.st_ino;
                u64[i++] = st.st_size;
                u64[i++] = st.st_blocks;

                uint32_t *u32 = (uint32_t*)(u64 + i);
                i = 0;
                u32[i++] = st.st_dev;
                u32[i++] = st.st_mode;
                u32[i++] = st.st_nlink;
                u32[i++] = st.st_uid;
                u32[i++] = st.st_gid;
                u32[i++] = st.st_rdev;
                u32[i++] = st.st_blksize;
                ((int32_t*)(u32))[i++] = 0;

                volatileReqs[i].result_end = req.result_start + 104;
            } else if (ok == 0) {
                volatileReqs[i].result_end = 0;
            } else {
                *((int32_t*)(toGPUBuf + req.result_start + 100)) = errno;
                errno = 0;
            }
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_DLOPEN) {
            // Load shared library
            void *lib = dlopen(filename, RTLD_LAZY);
            *((int64_t*)(toGPUBuf + req.result_start)) = (int64_t)lib;
            volatileReqs[i].result_end = req.result_start + 8;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_DLCALL) {
            // Call a function in a shared library
            void *lib = (void*)req.offset;
            app->readFromGPUIO(req.result_start, req.count);
            void (*func)(void* src, uint32_t srcLength, void* dst, uint32_t dstLength);
            *(void **) (&func) = dlsym(lib, filename);
            (*func)((void*)(fromGPUBuf + req.result_start), int32_t(req.count), (void*)(toGPUBuf + req.data2_start), req.data2_end-req.data2_start);
            volatileReqs[i].result_end = 0;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_DLCLOSE) {
            // Close a shared library
            dlclose((void*)req.offset);
            req.status = IO_COMPLETE;

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

        } else if (req.ioType == IO_CHROOT) {
            // Chdir to given dir and make it root
            int result = chdir(filename);
            if (result == 0) result = chroot(".");
            volatileReqs[i].result_start = result;
            volatileReqs[i].result_end = 0;
            req.status = IO_COMPLETE;

        } else if (req.ioType == IO_EXIT) {
            req.status = IO_COMPLETE;
            exit(req.offset); // Uh..

        } else {
            req.status = IO_ERROR;

        }
        volatileReqs[i].status = req.status;
        if (verbose) fprintf(stderr, "IO completed: %d - status %d\n", i, volatileReqs[i].status);

        if (threadIdx >= 0) {
            std::lock_guard<std::mutex> guard(completed_mutex);
            completed[threadIdx] = true;
        }
    }

    static void handleIORequests(ComputeApplication *app, bool verbose, ioRequests *ioReqs, char *toGPUBuf, char *fromGPUBuf, volatile bool *ioRunning, volatile bool *ioReset) {

        int threadCount = 64;
        std::thread threads[threadCount];
        int threadIdx = 0;
        volatile bool *completed = (volatile bool*)malloc(sizeof(bool) * threadCount);
        auto volatileReqs = ((volatile ioRequest*)(ioReqs->requests));

        std::map<int, FILE*> fileCache{};

        int32_t lastReqNum = 0;
        if (verbose) fprintf(stderr, "IO Running\n");
        while (*ioRunning) {
            if (*ioReset) {
                while (threadIdx > 0) threads[--threadIdx].join();
                for (int i=0; i < threadCount; i++) completed[i] = 0;
                lastReqNum = 0;
                ioReqs->ioCount = 0;
                *ioReset = false;
            }
            int32_t reqNum = ((volatile ioRequests*)ioReqs)->ioCount % IO_REQUEST_COUNT;
            if (reqNum < lastReqNum) {
                for (int32_t i = lastReqNum; i < IO_REQUEST_COUNT; i++) {
                    if (verbose) fprintf(stderr, "Got IO request %d\n", i);
                    if (volatileReqs[i].ioType == IO_READ) {
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
                        threads[tidx] = std::thread(handleIORequest, app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, tidx, &fileCache);
                    } else {
                        handleIORequest(app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, -1, &fileCache);
                    }
                }
                lastReqNum = 0;
            }
            for (int32_t i = lastReqNum; i < reqNum; i++) {
                if (verbose) fprintf(stderr, "Got IO request %d\n", i);
                if (volatileReqs[i].ioType == IO_READ) {
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
                    threads[tidx] = std::thread(handleIORequest, app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, tidx, &fileCache);
                } else {
                    handleIORequest(app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, -1, &fileCache);
                }

            }
            lastReqNum = reqNum;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        while (threadIdx > 0) threads[--threadIdx].join();
        for (int i=0; i < threadCount; i++) completed[i] = false;
        if (verbose) fprintf(stderr, "Exited IO thread\n");
    }

};


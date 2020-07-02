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
        timeIval("Create instance");

        findPhysicalDevice();
        timeIval("Find device");

        createDevice();
        timeIval("Create device");


        // Create input and output buffers
        createBuffer();
        timeIval("Create buffers");

        createDescriptorSetLayout();
        createDescriptorSet();

        createComputePipeline();

        timeIval("Create compute pipeline");

        createCommandBuffer();

        timeIval("Create command buffer");

        createFence();
        mapMemory();

        timeIval("Map memory");

        char *ioHeapBuf = (char *)mappedIOHeapMemory;
        ioRequests *ioReqs = (ioRequests *)mappedIOMemory;
        char *heapBuf = (char *)mappedIOHeapMemory;
        int32_t *i32HeapBuf = (int32_t *)heapBuf;

        // Copy argv to the IO heap.
        int32_t heapEnd = heapGlobalsOffset;
        int32_t i32HeapEnd = heapGlobalsOffset / 4;
        int32_t i32_ptr = 0;
        i32HeapBuf[i32_ptr++] = i32HeapEnd + 2;            // start index of the argv array
        i32HeapBuf[i32_ptr++] = i32HeapEnd + 2 + argc * 2; // end index of the argv array
        int32_t heap_ptr = 4 * (2 + argc * 2);
        int32_t heapPtr = heapEnd + heap_ptr;
        int32_t i32HeapPtr = i32HeapEnd + 2;
        for (int i = 0; i < argc; i++) {
            int32_t len = strlen(argv[i]);
            i32HeapBuf[i32_ptr++] = heapPtr;          // start index of argv[i] on the heap
            i32HeapBuf[i32_ptr++] = heapPtr + len;    // end index of argv[i]
            memcpy(heapBuf + heap_ptr, argv[i], len); // copy argv[i] to the heap
            heap_ptr += len;
            heapPtr += len;
        }
        writeIOHeapToGPU(0, heap_ptr);

        timeIval("Write argv to GPU");

        ioReqs->ioCount = 0;
        ioReqs->programReturnValue = 0;
        ioReqs->maxIOCount = IO_REQUEST_COUNT;

        std::thread ioThread(handleIORequests, this, verbose, ioReqs, ioHeapBuf, &ioRunning, &ioReset);

        timeIval("Start IO thread");

        runProgram();

        timeIval("Run program");

        ioRunning = false;
        ioThread.join();

        exitCode = ioReqs->programReturnValue;

        timeIval("Join IO thread");

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
        vkMapMemory(device, ioRequestsMemory, 0, ioRequestsBufferSize, 0, &mappedIOMemory);
        vkMapMemory(device, ioHeapMemory, 0, ioHeapBufferSize, 0, &mappedIOHeapMemory);
    }

    void unmapMemory()
    {
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
    void readIOHeapFromGPU(VkDeviceSize offset, VkDeviceSize size) { readFromGPU(ioHeapMemory, offset, size); }

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

        //VkPhysicalDeviceFeatures deviceFeatures = {};
        std::vector<const char*> extensions;
        extensions.push_back(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME);

        VkPhysicalDeviceFeatures2 deviceFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };

        VkDeviceCreateInfo deviceCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .enabledLayerCount = 0, //static_cast<uint32_t>(enabledLayers.size()),
            .ppEnabledLayerNames = nullptr, //enabledLayers.data(),
            .pQueueCreateInfos = &queueCreateInfo,
            .queueCreateInfoCount = 1,
            //.pEnabledFeatures = &deviceFeatures,
            .pNext = &deviceFeatures,
            //.enabledExtensionCount = (uint32_t)extensions.size(),
            //.ppEnabledExtensionNames = extensions.data(),
        };

        VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); // create logical device.

        // Get a handle to the only member of the queue family.
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
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
        createAndAllocateBuffer(&heapBuffer, heapBufferSize, &heapMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
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
        if (code != NULL) {
            free(code);
            code = NULL;
        }
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
        delete[] code;

        timeIval("vkCreateShaderModule");

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

        /*
        Now, we finally create the compute pipeline.
        */
        VK_CHECK_RESULT(
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline));

        timeIval("vkCreateComputePipelines");

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
        Copy the staging buffers for global data to the heap.
        */
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = heapGlobalsOffset;
        copyRegion.size = heapSize;
        vkCmdCopyBuffer(commandBuffer, ioHeapBuffer, heapBuffer, 1, &copyRegion);

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
        vkFreeMemory(device, heapMemory, NULL);
        vkFreeMemory(device, ioRequestsMemory, NULL);
        vkFreeMemory(device, ioHeapMemory, NULL);
        vkDestroyBuffer(device, heapBuffer, NULL);
        vkDestroyBuffer(device, ioRequestsBuffer, NULL);
        vkDestroyBuffer(device, ioHeapBuffer, NULL);
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
    }

    static FILE* openFile(char *filename, FILE* file, const char *mode) {
        if (file != NULL) return file;
        return fopen(filename, mode);
    }

    static void closeFile(FILE* file) {
        if (file != NULL) fclose(file);
    }

    static void handleIORequest(ComputeApplication *app, bool verbose, ioRequests *ioReqs, char *heapBuf, int i, volatile bool *completed, int threadIdx) {
        while (ioReqs->requests[i].status != IO_START);
        ioRequest req = ioReqs->requests[i];
        if (verbose) printf("IO req %d: t:%d s:%d off:%ld count:%ld fn_start:%d fn_end:%d res_start:%d res_end:%d\n", i, req.ioType, req.status, req.offset, req.count, req.filename_start, req.filename_end, req.result_start, req.result_end);
        char *filename = NULL;
        FILE *file = NULL;
        if (req.filename_end == 0) {
            if (req.filename_start == 0) {
                file = stdin;
            } else if (req.filename_start == 1) {
                file = stdout;
            } else if (req.filename_start == 2) {
                file = stderr;
            }
        } else {
            VkDeviceSize filenameLength = req.filename_end - req.filename_start;
            if (verbose) printf("Filename length %lu\n", filenameLength);
            filename = (char*)calloc(filenameLength + 1, 1);
            memcpy(filename, heapBuf + req.filename_start, filenameLength);
            filename[filenameLength] = 0;
        }
        if (req.ioType == IO_READ) {
            if (verbose) printf("Read %s\n", filename);
            FILE *fd = openFile(filename, file, "r");
            fseek(fd, req.offset, SEEK_SET);
            int32_t bytes = fread(heapBuf + req.result_start, 1, req.count, fd);
            if (file == NULL) closeFile(fd);
            req.result_end = req.result_start + bytes;
            req.status = IO_COMPLETE;
            ioReqs->requests[i] = req;
            if (verbose) printf("IO completed: %d - status %d\n", i, ioReqs->requests[i].status);
        } else if (req.ioType == IO_WRITE) {
            FILE *fd = openFile(filename, file, "r+");
            if (req.offset < 0) {
                fseek(fd, -1-req.offset, SEEK_END);
            } else {
                fseek(fd, req.offset, SEEK_SET);
            }
            int bytes = fwrite(heapBuf + req.result_start, 1, req.count, fd);
            if (file == NULL) closeFile(fd);
            req.result_end = req.result_start + bytes;
            req.status = IO_COMPLETE;
            ioReqs->requests[i] = req;
            if (verbose) printf("IO completed: %d - status %d\n", i, ioReqs->requests[i].status);
        } else if (req.ioType == IO_CREATE) {
            FILE *fd = openFile(filename, file, "w");
            int bytes = fwrite(heapBuf + req.result_start, 1, req.count, fd);
            if (file == NULL) closeFile(fd);
            ioReqs->requests[i].result_end = req.result_start + bytes;
            ioReqs->requests[i].status = IO_COMPLETE;
        } else if (req.ioType == IO_DELETE) {
            remove(filename);
            ioReqs->requests[i].status = IO_COMPLETE;
        } else if (req.ioType == IO_TRUNCATE) {
            truncate(filename, req.count);
            ioReqs->requests[i].status = IO_COMPLETE;
        } else {
            ioReqs->requests[i].status = IO_ERROR;
        }
        if (threadIdx > 0) {
            std::lock_guard<std::mutex> guard(completed_mutex);
            completed[threadIdx] = true;
        }
    }

    static void handleIORequests(ComputeApplication *app, bool verbose, ioRequests *ioReqs, char *heapBuf, volatile bool *ioRunning, volatile bool *ioReset) {

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
                    threads[tidx] = std::thread(handleIORequest, app, verbose, ioReqs, heapBuf, i, completed, tidx);
                } else {
                    handleIORequest(app, verbose, ioReqs, heapBuf, i, completed, -1);
                }
            }
            lastReqNum = reqNum;
        }
        while (threadIdx > 0) threads[--threadIdx].join();
        for (int i=0; i < threadCount; i++) completed[i] = false;
        if (verbose) printf("Exited IO thread\n");
    }

};


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

struct ioRequest {
    int32_t ioType;
    int32_t filename_start;
    int32_t filename_end;
    int32_t offset;
    int32_t count;
    int32_t result_start;
    int32_t result_end;
    int32_t status;
}

#define IO_REQUEST_COUNT 8192

struct ioRequests {
    int32_t count;
    ioRequest requests[IO_REQUEST_COUNT];
}


/*
The application launches a compute shader that reads its inputs from stdin,
runs the compute kernel, and writes the result to stdout.

The data read from stdin is preceded by a 16-byte header:
    bytes 0..3: uint32_t outputBufferByteLength
    bytes 4..7: uint32_t vulkanDeviceIndex
    bytes 8..19: uint32_t workGroupSize[3]

The rest of the data is written to the buffer bound to layout binding 0.
The output buffer is bound to layout binding 1, and has the size outputBufferByteLength.
*/
class ComputeApplication
{
  private:
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

    VkBuffer buffer;
    VkBuffer buffer2;

    VkDeviceMemory bufferMemory;
    VkDeviceMemory bufferMemory2;

    VkBuffer inputBuffer;
    VkDeviceMemory inputBufferMemory;

	// Internal buffers for the device.
    VkBuffer heapBuffer;
    VkDeviceMemory heapBufferMemory;

    VkBuffer ioBuffer;
    VkDeviceMemory ioBufferMemory;

    uint32_t bufferSize; // size of `buffer` in bytes.
    uint32_t inputBufferSize; // size of `inputBuffer` in bytes.
    uint32_t heapBufferSize; // size of `heapBuffer` in bytes.
    uint32_t ioBufferSize; // size of `heapBuffer` in bytes.

    std::vector<const char *> enabledLayers;

    VkQueue queue; // a queue supporting compute operations.

    VkFence fence;

    void *mappedInputMemory = NULL;
    void *mappedHeapMemory = NULL;
    void *mappedIoMemory = NULL;
    void *mappedOutputMemory = NULL;
    void *mappedOutputMemory2 = NULL;

    uint32_t queueFamilyIndex;

    uint32_t vulkanDeviceIndex = 0;

    uint32_t maxRequestSize = 1024;
    uint32_t maxResponseSize = 1024 * 4;
    uint32_t heapSize = 8192 * 4;
    uint32_t ioSize = sizeof(ioRequests);

    uint32_t workSize[3] = {16, 1, 1};
    char *input;
    char *heap;

    ioRequests *ioReqs;

    const char *programFileName;

    uint32_t requestCount = workSize[0] * workSize[1] * workSize[2] * 1;

  public:
    void run(const char *fileName)
    {

        programFileName = fileName;

#ifdef WIN32
        _setmode(_fileno(stdout), _O_BINARY);
        _setmode(_fileno(stdin), _O_BINARY);
#endif

        inputBufferSize = maxRequestSize * requestCount;
        bufferSize = maxResponseSize * requestCount;
        heapBufferSize = heapSize * requestCount;
        ioBufferSize = ioSize;

        input = (char *)malloc(inputBufferSize);
        heap = (char *)malloc(heapBufferSize);

        // Initialize vulkan:
        createInstance();
        findPhysicalDevice();
        createDevice();

        // Create input and output buffers
        createBuffer();

        createDescriptorSetLayout();
        createDescriptorSet();

        createComputePipeline();
        createCommandBuffer();

        createFence();
        mapMemory();

        bool firstFrame = true;
        bool ioRunning = true;

        std::thread ioThread([](ioRequests *ioReqs) {
            int32_t lastReqNum = 0;
            while (ioRunning) {
                int32_t reqNum = ioReqs->count;
                for (int32_t i = lastReqNum; i < reqNum; i++) {
                    // handleIORequest(ioReqs, i);
                    ioRequest req = ioReqs.requests[i];
                    char* filename = malloc(req.filename_end - req.filename_start + 1);
                    readHeapFromGPU(req.filename_start, req.filename_end - req.filename_start);
                    memcpy(filename, mappedHeapMemory + req.filename_start, req.filename_end - req.filename_start);
                    filename[req.filename_end - req.filename_start] = 0;
                    if (req.type == IO_READ) {
                        ioReqs.requests[i].status = IO_IN_PROGRESS;
                        int fd = fopen(filename, "r");
                        fseek(fd, req.offset, SEEK_SET);
                        int32_t bytes = fread(mappedHeapMemory + req.result_start, 1, req.count, fd);
                        fclose(fd);
                        writeHeapToGPU(req.result_start, bytes);
                        ioReqs.requests[i].result_end = req.result_start + bytes;
                        ioReqs.requests[i].status = IO_COMPLETE;
                    } else if (req.type == IO_WRITE) {
                        ioReqs.requests[i].status = IO_IN_PROGRESS;
                        int fd = fopen(filename, "r+");
                        fseek(fd, req.offset, SEEK_SET);
                        int bytes = fwrite(mappedHeapMemory + req.result_start, 1, req.count, fd);
                        fclose(fd);
                        ioReqs.requests[i].result_end = req.result_start + bytes;
                        ioReqs.requests[i].status = IO_COMPLETE;
                    } else if (req.type == IO_CREATE) {
                        ioReqs.requests[i].status = IO_IN_PROGRESS;
                        int fd = fopen(filename, "w");
                        fclose(fd);
                        ioReqs.requests[i].status = IO_COMPLETE;
                    } else if (req.type == IO_DELETE) {
                        ioReqs.requests[i].status = IO_IN_PROGRESS;
                        remove(filename);
                        ioReqs.requests[i].status = IO_COMPLETE;
                    } else if (req.type == IO_TRUNCATE) {
                        ioReqs.requests[i].status = IO_IN_PROGRESS;
                        truncate(filename, req.count);
                        ioReqs.requests[i].status = IO_COMPLETE;
                    } else {
                        ioReqs.requests[i].status = IO_ERROR;
                    }
                }
                lastReqNum = reqNum;
            }
        };, ioReqs);

        writeHeap();

            writeInput();
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; i++) {
            startCommandBuffer();
            waitCommandBuffer();
            //swapOutputBuffers();
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            readOutput();
            writeOutput();


        printf("\nElapsed: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
        printf("Test runs per second: %.0f\n\n", (requestCount * 100.0) / (0.001 * std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()));

        unmapMemory();
        cleanup();
    }

    void swapOutputBuffers() {
        auto tmpB = buffer;
        buffer = buffer2;
        buffer2 = tmpB;
        auto tmp = bufferMemory;
        bufferMemory = bufferMemory2;
        bufferMemory2 = tmp;
        auto tmpM = mappedOutputMemory;
        mappedOutputMemory = mappedOutputMemory2;
        mappedOutputMemory2 = tmpM;
    }

    void mapMemory()
    {
        // Map the buffer memory, so that we can read from it on the CPU.
        vkMapMemory(device, inputBufferMemory, 0, inputBufferSize, 0, &mappedInputMemory);
        vkMapMemory(device, heapBufferMemory, 0, heapBufferSize, 0, &mappedHeapMemory);
        vkMapMemory(device, ioBufferMemory, 0, ioBufferSize, 0, &mappedIoMemory);
        ioReqs = (ioRequests *)mappedIoMemory;
        vkMapMemory(device, bufferMemory, 0, bufferSize, 0, &mappedOutputMemory);
        vkMapMemory(device, bufferMemory2, 0, bufferSize, 0, &mappedOutputMemory2);
    }

    void unmapMemory()
    {
        vkUnmapMemory(device, inputBufferMemory);
        vkUnmapMemory(device, heapBufferMemory);
        vkUnmapMemory(device, ioBufferMemory);
        vkUnmapMemory(device, bufferMemory);
        vkUnmapMemory(device, bufferMemory2);
    }


    void writeHeapToGPU(int offset, int size)
    {
        VkMappedMemoryRange heapBufferMemoryRange = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = NULL,
            .memory = heapBufferMemory,
            .offset = offset,
            .size = size};
        vkFlushMappedMemoryRanges(device, 1, &heapBufferMemoryRange);
    }

    void readHeapFromGPU(int offset, int size)
    {
        VkMappedMemoryRange heapBufferMemoryRange = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = NULL,
            .memory = heapBufferMemory,
            .offset = offset,
            .size = size};
        vkInvalidateMappedMemoryRanges(device, 1, &heapBufferMemoryRange);
    }

    void writeHeap()
    {
        // vkMapMemory(device, heapBufferMemory, 0, heapBufferSize, 0, &mappedHeapMemory);
        memcpy(mappedHeapMemory, heap, heapBufferSize);

        VkMappedMemoryRange heapBufferMemoryRange = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = NULL,
            .memory = heapBufferMemory,
            .offset = 0,
            .size = VK_WHOLE_SIZE};

        // Flush the CPU memory to the input buffer.
        vkFlushMappedMemoryRanges(device, 1, &heapBufferMemoryRange);
        // vkUnmapMemory(device, heapBufferMemory);
    }

    void writeInput()
    {
        // vkMapMemory(device, inputBufferMemory, 0, inputBufferSize, 0, &mappedInputMemory);
        memcpy(mappedInputMemory, input, inputBufferSize);

        VkMappedMemoryRange inputBufferMemoryRange = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = NULL,
            .memory = inputBufferMemory,
            .offset = 0,
            .size = VK_WHOLE_SIZE};

        // Flush the CPU memory to the input buffer.
        vkFlushMappedMemoryRanges(device, 1, &inputBufferMemoryRange);
        // vkUnmapMemory(device, inputBufferMemory);
    }


    void readOutput()
    {
        // vkMapMemory(device, bufferMemory, 0, bufferSize, 0, &mappedOutputMemory);
        VkMappedMemoryRange bufferMemoryRange = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .pNext = NULL,
            .memory = bufferMemory,
            .offset = 0,
            .size = VK_WHOLE_SIZE};

        // Flush the output buffer memory to the CPU.
        vkInvalidateMappedMemoryRanges(device, 1, &bufferMemoryRange);
    }

    void writeOutput()
    {
    	for (int j = 0; j < requestCount; j++) {
	    	bool allOk = true;
	        for (int i = 0; i < 1024; i++) {
	            int ok = ((int*)mappedOutputMemory)[i + j*1024];
	            if (ok == 0) break;
	            if (ok != 1) {
	            	printf("[%d] Test %d failed: %d\n", j, i, ok);
	            	allOk = false;
	            }
	        }
	        if (allOk) {
	        	printf("[%d] All tests succeeded.\n", j);
	        }
        }
        fflush(stdout);
        // vkUnmapMemory(device, bufferMemory);
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

        VkPhysicalDeviceFeatures deviceFeatures = {};

        VkDeviceCreateInfo deviceCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .enabledLayerCount = static_cast<uint32_t>(enabledLayers.size()),
            .ppEnabledLayerNames = enabledLayers.data(),
            .pQueueCreateInfos = &queueCreateInfo,
            .queueCreateInfoCount = 1,
            .pEnabledFeatures = &deviceFeatures,
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

    void createAndAllocateBuffer(VkBuffer *buffer, uint32_t bufferSize, VkDeviceMemory *bufferMemory, VkMemoryPropertyFlagBits flags)
    {
        VkBufferCreateInfo bufferCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bufferSize,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
        createAndAllocateBuffer(&buffer, bufferSize, &bufferMemory, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        createAndAllocateBuffer(&buffer2, bufferSize, &bufferMemory2, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        createAndAllocateBuffer(&inputBuffer, inputBufferSize, &inputBufferMemory, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        createAndAllocateBuffer(&heapBuffer, heapBufferSize, &heapBufferMemory, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        createAndAllocateBuffer(&ioBuffer, ioBufferSize, &ioBufferMemory, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[5] = {
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
            {.binding = 4,
             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .descriptorCount = 1,
             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        };

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 5,
            .pBindings = descriptorSetLayoutBindings};

        // Create the descriptor set layout.
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
        //printf("createDescriptorSetLayout done\n");
    }

    void createDescriptorSet()
    {
        VkDescriptorPoolSize descriptorPoolSize = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 5};

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

        VkDescriptorBufferInfo inputDescriptorBufferInfo = {
            .buffer = inputBuffer,
            .offset = 0,
            .range = inputBufferSize};

        VkDescriptorBufferInfo outputDescriptorBufferInfo = {
            .buffer = buffer,
            .offset = 0,
            .range = bufferSize};

        VkDescriptorBufferInfo heapDescriptorBufferInfo = {
            .buffer = heapBuffer,
            .offset = 0,
            .range = heapBufferSize};

        VkDescriptorBufferInfo ioDescriptorBufferInfo = {
            .buffer = ioBuffer,
            .offset = 0,
            .range = ioBufferSize};

        VkWriteDescriptorSet writeDescriptorSets[5] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &inputDescriptorBufferInfo,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &outputDescriptorBufferInfo,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &heapDescriptorBufferInfo,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet,
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &ioDescriptorBufferInfo,
            },
        };

        vkUpdateDescriptorSets(device, 4, writeDescriptorSets, 0, NULL);
        //printf("vkUpdateDescriptorSets done\n");
    }

    void updateOutputDescriptorSet()
    {
        VkDescriptorBufferInfo outputDescriptorBufferInfo = {
            .buffer = buffer,
            .offset = 0,
            .range = bufferSize};

        VkWriteDescriptorSet writeDescriptorSets[1] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &outputDescriptorBufferInfo,
            }
        };

        vkUpdateDescriptorSets(device, 1, writeDescriptorSets, 0, NULL);
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

        long filesizepadded = long(ceil(filesize / 4.0)) * 4;

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

    void createComputePipeline()
    {
        /*
        We create a compute pipeline here.
        */

        /*
        Create a shader module. A shader module basically just encapsulates some shader code.
        */
        uint32_t filelength;
        // the code in comp.spv was created by running the command:
        // glslangValidator.exe -V shader.comp
        uint32_t *code = readFile(filelength, programFileName);
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pCode = code;
        createInfo.codeSize = filelength;

        VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));
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

        VkComputePipelineCreateInfo pipelineCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = shaderStageCreateInfo,
            .layout = pipelineLayout};

        /*
        Now, we finally create the compute pipeline.
        */
        VK_CHECK_RESULT(
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline));
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
        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000000));
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
        vkFreeMemory(device, bufferMemory, NULL);
        vkFreeMemory(device, inputBufferMemory, NULL);
        vkFreeMemory(device, heapBufferMemory, NULL);
        vkDestroyBuffer(device, buffer, NULL);
        vkDestroyBuffer(device, inputBuffer, NULL);
        vkDestroyBuffer(device, heapBuffer, NULL);
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
    }
};

int main(int argc, char *argv[])
{
    ComputeApplication app;

    try
    {
        app.run(argc > 1 ? argv[1] : "test_file.spv");
    }
    catch (const std::runtime_error &e)
    {
        printf("%s\n", e.what());
        app.cleanup();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

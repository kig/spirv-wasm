#include "compute_application.hpp"


class Grep : public ComputeApplication
{

public:

    Grep() {
        ioHeapSize = 4096;
        heapSize = 4096;

        workSize[0] = 100;

        timings = true;
    }

    void runProgram() {
        /*
        int n = 4;
        int len = n * 1 * 1048576;
        char *c = (char*)memalign(64, len);
        char *d = (char*)mappedIOHeapMemory;
        //char *d = (char*)memalign(64, len);
        memset(c, 5, len);
        std::thread threads[n];
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        int i;
        for (i = 0; i < 100; i++) {
            for (int j = 0; j < n; j++) threads[j] = std::thread(memcpy, d+(j*len/n), c+(j*len/n), len/n);
            for (int j = 0; j < n; j++) threads[j].join();
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        printf("%.2f GB/s\n\n", ((float)len * i) / 1e9 / (0.000001 * std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()));
        */

        startCommandBuffer();
        waitCommandBuffer();
    }
};

int main(int argc, char *argv[])
{
    Grep app;

    try
    {
        app.run("grep.spv", argc, argv);
    }
    catch (const std::runtime_error &e)
    {
        printf("%s\n", e.what());
        app.cleanup();
        return EXIT_FAILURE;
    }

    return app.exitCode;
}

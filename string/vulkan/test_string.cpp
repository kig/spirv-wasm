#include "compute_application.hpp"

class App : public ComputeApplication
{
  public:
    App() {
        heapSize = 8192;
        ioHeapSize = 1024;
        workSize[0] = 80;
        timings = true;
        runIO = false;
    }

    void runProgram() {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        int i;
        for (i = 0; i < 10; i++) {
            startCommandBuffer();
            waitCommandBuffer();
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    	for (int j = 0; j < threadCount; j++) {
	    	bool allOk = true;
	        for (int k = 0; k < ioHeapSize/4; k++) {
	            int ok = ((int*)mappedIOHeapMemory)[j*(ioHeapSize/4) + k];
	            if (ok == 0) break;
	            if (ok != 1) {
	            	printf("[%d] Test %d failed: %d\n", k, i, ok);
	            	allOk = false;
	            }
	        }
	        if (allOk) {
	        	// printf("[%d] All tests succeeded.\n", j);
	        }
        }

        printf("\nElapsed: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
        printf("Test runs per second: %.0f\n\n", (float)(threadCount * i) / (0.000001 * std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()));

    }
};

int main(int argc, char *argv[])
{
    App app;

    try
    {
        app.run("test_string.spv", argc, argv);
    }
    catch (const std::runtime_error &e)
    {
        printf("%s\n", e.what());
        app.cleanup();
        return EXIT_FAILURE;
    }

    return app.exitCode;
}

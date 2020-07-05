#include "compute_application.hpp"

class App : public ComputeApplication
{
  public:
    App() {
        workSize[0] = 4;
        verbose = false;
    }

    void runProgram() {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        int i;
        for (i = 0; i < 1; i++) {
            while(ioReset);
            startCommandBuffer();
            waitCommandBuffer();
            ioReset = true;
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        printf("\nElapsed: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
        printf("Test runs per second: %.0f\n\n", (float)(threadCount * i) / (0.000001 * std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()));
    }
};

int main(int argc, char *argv[])
{
    App app;

    try
    {
        app.run("test_file.spv", argc, argv);
    }
    catch (const std::runtime_error &e)
    {
        printf("%s\n", e.what());
        app.cleanup();
        return EXIT_FAILURE;
    }

    return app.exitCode;
}

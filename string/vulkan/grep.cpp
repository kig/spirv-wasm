#include "compute_application.hpp"

class Grep : public ComputeApplication
{

    public:
        Grep() {
            heapSize = 4096;

            workSize[0] = 100;
            threadCount = workSize[0] * workSize[1] * workSize[2] * 255;

            heapGlobalsOffset = heapSize * threadCount;

            timings = true;
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

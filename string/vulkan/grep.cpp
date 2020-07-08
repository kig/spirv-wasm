#include "compute_application.hpp"

class Grep : public ComputeApplication
{
public:
    Grep() {
        workSize[0] = 64;
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

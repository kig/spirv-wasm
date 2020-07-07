#include "compute_application.hpp"

int main(int argc, char *argv[])
{
    ComputeApplication app;
    app.workSize[0] = 20;
    app.timings = true;
    // app.verbose = true;

    try
    {
        app.run(argv[1], argc-1, argv+1);
    }
    catch (const std::runtime_error &e)
    {
        printf("%s\n", e.what());
        app.cleanup();
        return EXIT_FAILURE;
    }

    return app.exitCode;
}

#include "compute_application.hpp"

int usage() {
    printf("USAGE: gls [-t] [-v] program.(spv|glsl) args...\n");
    return EXIT_FAILURE;
}

int main(int argc, char *argv[])
{
    ComputeApplication app;
    app.workSize[0] = 20;
    int argIdx = 1;
    if (argIdx >= argc) return usage();
    if (strcmp(argv[argIdx], "-t") == 0) {
      app.timings = true;
      argIdx++;
    }
    if (argIdx >= argc) return usage();
    if (strcmp(argv[argIdx], "-v") == 0) {
      app.verbose = true;
      argIdx++;
    }
    if (argIdx >= argc) return usage();
    if (strcmp(argv[argIdx], "-t") == 0) {
      app.timings = true;
      argIdx++;
    }
    if (argIdx >= argc) return usage();

    int len = strlen(argv[argIdx]);
    if (len > 5 && strcmp(argv[argIdx] + (len-5), ".glsl") == 0) {
        char cmd[len*2 + 20];
        argv[argIdx][len-5] = 0;
        sprintf(cmd, "glsl2spv \"%s.glsl\" \"%s.spv\"", argv[argIdx], argv[argIdx]);
        system(cmd);
        argv[argIdx][len-5] = '.';
        argv[argIdx][len-4] = 's';
        argv[argIdx][len-3] = 'p';
        argv[argIdx][len-2] = 'v';
        argv[argIdx][len-1] = 0;
    }

    try
    {
        app.run(argv[argIdx], argc-argIdx, argv+argIdx);
    }
    catch (const std::runtime_error &e)
    {
        printf("%s\n", e.what());
        app.cleanup();
        return EXIT_FAILURE;
    }

    return app.exitCode;
}

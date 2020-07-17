#include "compute_application.hpp"

#include <stdio.h>
#include <ctype.h>
#include <sys/stat.h>

#define MD5_LEN 32

bool getFileMD5(char *filename, char *md5sum)
{
    char cmd[500];
    snprintf(cmd, sizeof(cmd), "md5sum %s 2>/dev/null", filename);

    FILE *pipe = popen(cmd, "r");
    if (pipe == NULL) return false;
    size_t bytes = fread(md5sum, 1, MD5_LEN, pipe);
    pclose(pipe);

    md5sum[bytes] = 0;
    return bytes == MD5_LEN;
}

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
        char cmd[len + 520];
        system("mkdir -p ~/.gls/cache/");
        char md5sum[MD5_LEN + 1];
        if (!getFileMD5(argv[argIdx], md5sum)) {
        	fprintf(stderr, "Failed to get MD5 sum of file\n");
        	return EXIT_FAILURE;
        }
        char spvFilename[500];
        snprintf(spvFilename, sizeof(spvFilename), "%s/.gls/cache/%s.spv", getenv("HOME"), md5sum);
        struct stat st;
        if (0 != stat(spvFilename, &st)) {
        	fprintf(stderr, "Compiling to SPIR-V\n");
	        snprintf(cmd, sizeof(cmd), "glsl2spv \"%s\" %s", argv[argIdx], spvFilename);
	        system(cmd);
	        if (0 != stat(spvFilename, &st)) {
	        	fprintf(stderr, "Failed to compile SPIR-V: %s\n", cmd);
		        return EXIT_FAILURE;
	        }
        }
        argv[argIdx] = spvFilename;
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

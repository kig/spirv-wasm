#define TAG(v) (((v>>24) & 0xff) | ((v>>8) & 0xff00) | (((uint32_t)v<<8) & 0xff0000) | (((uint32_t)v << 24) & 0xff000000))

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
        #ifndef NDEBUG
        fprintf(stderr, "Op: %8x OpCode: %d WordCount:%d\n", op, opCode, wordCount);
        #endif
        int j = i+1;
        if (opCode == 16) { // OpExecutionMode
            uint32_t entryPoint = code[j++];
            uint32_t mode = code[j++];
            if (verbose) fprintf(stderr, "EntryPoint: %d Mode: %d\n", entryPoint, mode);
            if (mode == 17) { // LocalSize
                localSize[0] = code[j++];
                localSize[1] = code[j++];
                localSize[2] = code[j++];
                if (verbose) fprintf(stderr, "LocalSize: %d %d %d\n", localSize[0], localSize[1], localSize[2]);
            }
        }
        if (opCode == 4) { // OpSourceExtension
            int j = i + 1;
            uint32_t tag = code[j++];
            if (verbose) fprintf(stderr, "OpSourceExtension tag %.4s %d\n", (char*)(&tag), code[j]);
            if (tag == TAG('glo=')) {
                const uint32_t len = 4 * (wordCount - 2);
                globalsLen += len;
                if (globals != NULL) globals = (char*)realloc(globals, globalsLen+1);
                else globals = (char *)malloc(globalsLen+1);
                memcpy(globals + (globalsLen-len), code + j, len);
                globals[globalsLen] = 0;
                if (verbose) fprintf(stderr, "globals[%d]: %s\n", globalsLen, globals);
            }
            else if (tag == TAG('tgc=')) workSize[0] = code[j++];
            else if (tag == TAG('ths=')) heapBufferSize = code[j++];
            else if (tag == TAG('tti=')) fromGPUBufferSize = code[j++];
            else if (tag == TAG('tfi=')) toGPUBufferSize = code[j++];
        }
        i += wordCount > 0 ? wordCount-1 : 0;
    }
}

void readShader() {
    code = readFile(filelength, programFileName);
    parseLocalSize(code);
}

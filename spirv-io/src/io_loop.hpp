static void signalIOProgress() {
    IOProgressCount++;
}

static FILE* openFile(char *filename, FILE* file, const char* flags) {
    if (file != NULL) return file;
    return fopen(filename, flags);
}

static void closeFile(FILE* file) {
    if (file != NULL) fclose(file);
}

static void handleIORequest(ComputeApplication *app, bool verbose, ioRequests *ioReqs, char *toGPUBuf, char *fromGPUBuf, int i, volatile bool *completed, int threadIdx, std::map<int, FILE*> *fileCache) {
    volatile ioRequest *volatileReqs = ((volatile ioRequest*)(ioReqs->requests));
    while (volatileReqs[i].status != IO_START) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    volatileReqs[i].status = IO_IN_PROGRESS;
    ioRequest req = ioReqs->requests[i];
    if (verbose) fprintf(stderr, "IO req %d: t:%d s:%d off:%ld count:%ld fn_start:%d fn_end:%d res_start:%d res_end:%d\n", i, req.ioType, req.status, req.offset, req.count, req.filename_start, req.filename_end, req.result_start, req.result_end);

    // Read filename from the GPU
    char *filename = NULL;
    FILE *file = NULL;
    if (req.filename_end == 0) {
        if (req.filename_start == 0) {
            file = stdin;
        } else if (req.filename_start == 1) {
            file = stdout;
        } else if (req.filename_start == 2) {
            file = stderr;
        } else {
            std::lock_guard<std::mutex> guard(fileCache_mutex);
            file = fileCache->at(req.filename_start);
        }
    } else if (req.filename_end == -2) {
    } else {
        int64_t filenameLength = req.filename_end - req.filename_start;
        if (verbose) fprintf(stderr, "Filename length %lu\n", filenameLength);
        filename = (char*)calloc(filenameLength + 1, 1);
        app->readFromGPUIO(req.filename_start, filenameLength);
        memcpy(filename, fromGPUBuf + req.filename_start, filenameLength);
        filename[filenameLength] = 0;
    }

    // Process IO command
    if (req.ioType == IO_READ) {
        if (verbose) fprintf(stderr, "Read %s\n", filename);
        auto fd = openFile(filename, file, "rb");
        fseek(fd, req.offset, SEEK_SET);

        int32_t bytes = 0;
        int64_t totalRead = 0;

        const int32_t compressionType = req.compression & 0xff000000;
        const int32_t compressionData = req.compression & 0x00ffffff;
        const int32_t blockSize = compressionData & 0x000fffff;
        int32_t compressionSpeed = ((compressionData & 0x00f00000) >> 20) - 1;
        bool autoCompress = compressionSpeed == -1;
        if (compressionSpeed == -1) compressionSpeed = 7;

        if (verbose) fprintf(stderr, "Compression: %08x, type: %d, speed: %d, blockSize: %d\n", req.compression, compressionType, compressionSpeed, blockSize);

        bool doUncompressedRead = true;

        if (compressionType != 0) {

            doUncompressedRead = false;

            if (compressionType == IO_COMPRESS_ZSTD) {
                size_t const buffInSize = ZSTD_CStreamInSize();
                void*  const buffIn  = malloc(buffInSize);
                size_t const buffOutSize = ZSTD_CStreamOutSize();
                void*  const buffOut = malloc(buffOutSize);

                int minLevel = ZSTD_minCLevel();
                int maxLevel = ZSTD_maxCLevel();
                float speedRatio = sqrt(compressionSpeed/9.0);
                int level = round(maxLevel * (1-speedRatio) + minLevel * speedRatio);

                ZSTD_CCtx* const cctx = ZSTD_createCCtx();
                ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
                ZSTD_CCtx_setParameter(cctx, ZSTD_c_strategy, ZSTD_fast);

                size_t readCount = req.count;
                size_t toRead = readCount < buffInSize ? readCount : buffInSize;
                for (;;) {
                    size_t read = fread(buffIn, 1, readCount < toRead ? readCount : toRead, fd);
                    readCount -= read;
                    totalRead += read;
                    bool lastChunk = (read < toRead || readCount == 0);
                    ZSTD_EndDirective const mode = lastChunk ? ZSTD_e_end : ZSTD_e_continue;
                    ZSTD_inBuffer input = { buffIn, read, 0 };
                    bool finished = false;
                    do {
                        ZSTD_outBuffer output = { buffOut, buffOutSize, 0 };
                        size_t const remaining = ZSTD_compressStream2(cctx, &output, &input, mode);
                        memcpy(toGPUBuf + req.result_start + bytes, buffOut, output.pos);
                        bytes += output.pos;
                        finished = lastChunk ? (remaining == 0) : (input.pos == input.size);
                        volatileReqs[i].progress = bytes;
                        signalIOProgress();
                    } while (!finished);
                    if (lastChunk) break;
                }
                if (verbose) fprintf(stderr, "Wrote %ld => %d bytes\n", req.count, bytes);

                ZSTD_freeCCtx(cctx);
                free(buffIn);
                free(buffOut);
                req.status = IO_COMPLETE;

            } else if (compressionType == IO_COMPRESS_LZ4_BLOCK || compressionType == IO_COMPRESS_LZ4_BLOCK_STREAM) {
                // Use LZ4 block compression

                LZ4_stream_t lz4Stream_body;
                LZ4_stream_t* lz4Stream = &lz4Stream_body;

                const int64_t BLOCK_BYTES = 8192;

                char inpBuf[2][BLOCK_BYTES + 8];
                char cmpBuf[LZ4_COMPRESSBOUND(BLOCK_BYTES) + 8];
                int64_t inpBufIndex = 0;

                LZ4_initStream(lz4Stream, sizeof (*lz4Stream));

                int64_t readCount = req.count;

                if (compressionType == IO_COMPRESS_LZ4_BLOCK_STREAM) { // LZ4_BLOCK_STREAM

                    int32_t blockOff = 0;
                    int32_t blockBytes = 0;

                    const int32_t blockCount = 128;
                    bytes = 128 * 4;

                    for (int i = 0; i < blockCount; i++) {
                        *(int32_t*)(toGPUBuf + req.result_start + i*4) = 0;
                    }

                    if (verbose) fprintf(stderr, "Compression frame size: %d, internal block size: %ld\n", blockSize, BLOCK_BYTES);
                    if (verbose) fprintf(stderr, "Compression speed: %d\n", compressionSpeed);

                    while (true) {
                        char* const inpPtr = inpBuf[inpBufIndex];
                        const int64_t inpBytes = (int) fread(inpPtr, 1, std::min(BLOCK_BYTES, (readCount - totalRead)), fd);
                        totalRead += inpBytes;
                        if (0 == inpBytes) break;
                        int64_t cmpBytes = LZ4_compress_fast_continue(lz4Stream, inpPtr, cmpBuf, inpBytes, sizeof(cmpBuf), compressionSpeed);
                        if (cmpBytes <= 0) break;

                        if (totalRead == inpBytes && autoCompress) {
                            float compressionRatio = (float)cmpBytes / inpBytes;
                            if (blockOff == 0) {
                                if (compressionRatio > 0.7) {
                                    // Do uncompressed read
                                    memcpy(toGPUBuf + req.result_start, inpPtr, inpBytes);
                                    volatileReqs[i].compression = 0;
                                    volatileReqs[i].progress = inpBytes;
                                    signalIOProgress();
                                    bytes = inpBytes;
                                    doUncompressedRead = true;
                                    break;
                                } else if (compressionRatio < 0.3) {
                                    // Use stronger compression
                                    compressionSpeed = 3;
                                    // Recompress first block?
                                    // LZ4_resetStream_fast(lz4Stream);
                                    // cmpBytes = LZ4_compress_fast_continue(lz4Stream, inpPtr, cmpBuf, inpBytes, sizeof(cmpBuf), compressionSpeed);
                                } else {
                                    // Use faster compression
                                    compressionSpeed = 7;
                                }
                            }
                        }

                        memcpy(toGPUBuf + req.result_start + bytes, cmpBuf, cmpBytes);
                        bytes += cmpBytes;
                        blockBytes += cmpBytes;
                        if ((totalRead & (blockSize-1)) == 0) {
                            *(uint32_t*)(toGPUBuf + req.result_start + blockOff) = blockBytes;
                            if (verbose) fprintf(stderr, "%d\n", blockBytes);
                            blockBytes = 0;
                            blockOff += 4;
                            volatileReqs[i].progress = bytes;
                            signalIOProgress();
                            LZ4_resetStream_fast(lz4Stream);
                        }
                        inpBufIndex = (inpBufIndex + 1) % 2;
                    }
                    if (!doUncompressedRead) {
                         *(int32_t*)(toGPUBuf + req.result_start + blockOff) = blockBytes;
                        if (verbose && blockBytes > 0) fprintf(stderr, "[%d] = %d\n", req.result_start + blockOff, blockBytes);
                        if (verbose && totalRead > 0) fprintf(stderr, "Compression ratio: %.4f\n", (float)bytes/totalRead);
                        req.status = IO_COMPLETE;
                    }


                } else { // LZ4_BlOCK
                    while (true) {
                        char* const inpPtr = inpBuf[inpBufIndex];
                        const int inpBytes = (int) fread(inpPtr, 1, std::min(BLOCK_BYTES, (readCount - totalRead)), fd);
                        totalRead += inpBytes;
                        if (0 == inpBytes) break;
                        const int cmpBytes = LZ4_compress_fast_continue(lz4Stream, inpPtr, cmpBuf, inpBytes, sizeof(cmpBuf), compressionSpeed);
                        if (cmpBytes <= 0) break;
                        memcpy(toGPUBuf + req.result_start + bytes, cmpBuf, cmpBytes);
                        bytes += cmpBytes;
                        volatileReqs[i].progress = bytes;
                        signalIOProgress();
                        inpBufIndex = (inpBufIndex + 1) % 2;
                    }
                    if (verbose && totalRead > 0) fprintf(stderr, "Compression ratio: %.4f\n", (float)bytes/totalRead);
                    req.status = IO_COMPLETE;
                }


            } else if (compressionType == IO_COMPRESS_LZ4) { // LZ4_FRAME_STREAM

                if (verbose) fprintf(stderr, "IO_COMPRESS_LZ4\n");

                const size_t inChunkSize = 65536;

                const LZ4F_preferences_t kPrefs = {
                    { LZ4F_max256KB, LZ4F_blockLinked, LZ4F_noContentChecksum, LZ4F_frame,
                      0 /* unknown content size */, 0 /* no dictID */ , LZ4F_noBlockChecksum },
                     -3,   /* compression level; 0 == default */
                      0,   /* autoflush */
                      1,   /* favor decompression speed */
                      { 0, 0, 0 },  /* reserved, must be set to 0 */
                };

                LZ4F_compressionContext_t ctx;
                size_t const ctxCreation = LZ4F_createCompressionContext(&ctx, LZ4F_VERSION);
                void* const inBuff = malloc(inChunkSize);
                size_t const outCapacity = LZ4F_compressBound(inChunkSize, &kPrefs);
                void* const outBuff = malloc(outCapacity);

                if (!LZ4F_isError(ctxCreation) && inBuff && outBuff) {
                    uint64_t count_in = 0, count_out = 0;

                    assert(ctx != NULL);
                    assert(outCapacity >= LZ4F_HEADER_SIZE_MAX);
                    assert(outCapacity >= LZ4F_compressBound(inChunkSize, &kPrefs));

                    /* write frame header */
                    {   size_t const headerSize = LZ4F_compressBegin(ctx, outBuff, outCapacity, &kPrefs);
                        if (LZ4F_isError(headerSize)) {
                            fprintf(stderr, "Error: LZ4F Failed to start compression: error %u \n", (unsigned)headerSize);
                            exit(1);
                        }
                        memcpy(toGPUBuf + req.result_start + count_out, outBuff, headerSize);
                        count_out = headerSize;
                        volatileReqs[i].progress = count_out;
                        signalIOProgress();
                        if (verbose) fprintf(stderr, "Buffer size is %u bytes, header size %u bytes \n",
                                (unsigned)outCapacity, (unsigned)headerSize);
                    }

                    /* stream file */
                    for (;;) {
                        size_t const readSize = fread(inBuff, 1, std::min(inChunkSize, (req.count - count_in)), fd);
                        if (readSize == 0) break; /* nothing left to read from input file */
                        count_in += readSize;

                        size_t const compressedSize = LZ4F_compressUpdate(ctx,
                                                                outBuff, outCapacity,
                                                                inBuff, readSize,
                                                                NULL);
                        if (LZ4F_isError(compressedSize)) {
                            fprintf(stderr, "Error: LZ4F Compression failed: error %u \n", (unsigned)compressedSize);
                            exit(1);
                        }

                        if (verbose) fprintf(stderr, "Writing %u bytes\n", (unsigned)compressedSize);
                        memcpy(toGPUBuf + req.result_start + count_out, outBuff, compressedSize);
                        count_out += compressedSize;
                        volatileReqs[i].progress = count_out;
                        signalIOProgress();
                    }

                    /* flush whatever remains within internal buffers */
                    {   size_t const compressedSize = LZ4F_compressEnd(ctx,
                                                                outBuff, outCapacity,
                                                                NULL);
                        if (LZ4F_isError(compressedSize)) {
                            fprintf(stderr, "Error: LZ4F Failed to end compression: error %u\n", (unsigned)compressedSize);
                            exit(1);
                        }

                        if (verbose) fprintf(stderr, "Writing %u bytes \n", (unsigned)compressedSize);
                        memcpy(toGPUBuf + req.result_start + count_out, outBuff, compressedSize);
                        count_out += compressedSize;
                        volatileReqs[i].progress = count_out;
                        signalIOProgress();
                    }

                    bytes = count_out;
                    totalRead = count_in;
                    req.status = IO_COMPLETE;

                } else {
                    req.status = IO_ERROR;
                    fprintf(stderr, "Error: LZ4F resource allocation failed\n");
                    exit(1);
                }

                free(outBuff);
                free(inBuff);
                LZ4F_freeCompressionContext(ctx);   /* supports free on NULL */

            } else { // Unknown compression type
                req.status = IO_ERROR;
            }

        }

        if (doUncompressedRead) { // Uncompressed read.

            int64_t count = req.count;
            while (bytes < req.count) {
                int32_t readBytes = fread(toGPUBuf + req.result_start + bytes, 1, std::min(req.count-bytes, 65536L), fd);
                bytes += readBytes;
                volatileReqs[i].progress = bytes;
                signalIOProgress();
                if (readBytes == 0) break;
            }
            totalRead = (int64_t)bytes;
            if (verbose) fprintf(stderr, "Read %d = %ld bytes to GPU\n", bytes, totalRead);
            req.status = IO_COMPLETE;
        }

        if (file == NULL) { // Request reading file to page cache
            fseek(fd, 0, SEEK_END);
            readahead(fileno(fd), 0, ftell(fd));
        }
        if (file == NULL) closeFile(fd);

        volatileReqs[i].count = totalRead;
        volatileReqs[i].result_end = req.result_start + bytes;
        if (verbose) fprintf(stderr, "Sent count %ld = %ld to GPU\n", totalRead, volatileReqs[i].count);

    } else if (req.ioType == IO_WRITE) {
        struct stat st;
        int mode = 0;
        if (0 != stat(filename, &st)) {
            mode = 1;
        }
        errno = 0;
        if (verbose) fprintf(stderr, "openFile %s with mode %d\n", filename, mode);
        auto fd = openFile(filename, file, mode == 0 ? "rb+" : "w");
        app->readFromGPUIO(req.result_start, req.count);
        if (req.offset < 0) {
            fseek(fd, -1-req.offset, SEEK_END);
        } else {
            fseek(fd, req.offset, SEEK_SET);
        }
        if (verbose) fprintf(stderr, "write %p / %p: %.14s\n", fromGPUBuf, fromGPUBuf + req.result_start, fromGPUBuf + req.result_start);
        int32_t bytes = fwrite(fromGPUBuf + req.result_start, 1, req.count, fd);
        if (file == NULL) closeFile(fd);
        volatileReqs[i].result_end = req.result_start + bytes;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_READLINE) {
        if (file == NULL) {
            volatileReqs[i].result_end = 0;
            req.status = IO_ERROR;
        } else {
            auto fd = openFile(filename, file, "r");
            int64_t bytes = 0;
            while (bytes < req.count) {
                uint8_t ch = (uint8_t)fgetc(fd);
                if (ch != 10) {
                    toGPUBuf[req.result_start + bytes] = ch;
                    bytes++;
                } else {
                    break;
                }
            }
            volatileReqs[i].count = bytes;
            volatileReqs[i].result_end = req.result_start + (int32_t)bytes;
            req.status = IO_COMPLETE;
        }

    } else if (req.ioType == IO_CREATE) {
        // Create new file
        auto fd = openFile(filename, file, "wb");
        app->readFromGPUIO(req.result_start, req.count);
        int32_t bytes = fwrite(fromGPUBuf + req.result_start, 1, req.count, fd);
        if (file == NULL) closeFile(fd);
        volatileReqs[i].result_end = req.result_start + bytes;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_DELETE) {
        // Delete file
        remove(filename);
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_TRUNCATE) {
        // Truncate file
        truncate(filename, req.count);
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_NOP) {
        // Do nothing
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_PINGPONG) {
        // Send the received data back
        app->readFromGPUIO(req.result_start, req.count);
        memcpy(toGPUBuf + req.result_start, fromGPUBuf + req.offset, req.count);
        volatileReqs[i].result_end = req.result_start + req.count;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_RUN_CMD) {
        // Run command and copy the results to req.data
        int result = system(filename);
        volatileReqs[i].result_start = result;
        volatileReqs[i].result_end = 0;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_MOVE) {
        // Move file from filename to req.data
        char *dst = (char*)calloc(req.count+1, 1);
        app->readFromGPUIO(req.result_start, req.count);
        memcpy(dst, fromGPUBuf + req.result_start, req.count);
        dst[req.count] = 0;
        int result = rename(filename, dst);
        volatileReqs[i].result_start = result;
        volatileReqs[i].result_end = 0;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_COPY) {
        // Copy file from filename to req.data
        char *dst = (char*)calloc(req.count+1, 1);
        app->readFromGPUIO(req.result_start, req.count);
        memcpy(dst, fromGPUBuf + req.result_start, req.count);
        dst[req.count] = 0;
        std::error_code ec;
        std::filesystem::copy(filename, dst, ec);
        volatileReqs[i].result_start = ec.value();
        volatileReqs[i].result_end = 0;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_COPY_RANGE) {
        // Copy range[req.data, req.data + 8] of data from filename to (req.data + 16)
        req.status = IO_ERROR;

    } else if (req.ioType == IO_CD) {
        // Change directory
        int result = chdir(filename);
        volatileReqs[i].result_start = result;
        volatileReqs[i].result_end = 0;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_MKDIR) {
        // Make dir at filename
        int result = mkdir(filename, 0755);
        volatileReqs[i].result_start = result;
        volatileReqs[i].result_end = 0;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_RMDIR) {
        // Remove dir at filename
        int result = rmdir(filename);
        volatileReqs[i].result_start = result;
        volatileReqs[i].result_end = 0;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_GETCWD) {
        // Get current working directory
        char *s = getcwd(NULL, req.count);
        int32_t len = strlen(s);
        printf("getcwd: %s\n", s);
        memcpy(toGPUBuf + req.result_start, s, len);
        free(s);
        volatileReqs[i].result_end = req.result_start + len;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_OPEN) {
        // Open file
        FILE* fd = fopen(filename, req.offset == 0 ? "rb" : (req.offset == 1 ? "rb+" : (req.offset == 2 ? "rw" : "ra")));
        *(int32_t*)(toGPUBuf + req.result_start) = fileno(fd);
        volatileReqs[i].result_end = req.result_start + 4;
        std::lock_guard<std::mutex> guard(fileCache_mutex);
        fileCache->emplace(fileno(fd), fd);
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_CLOSE) {
        // Close open file
        if (file) {
            fclose(file);
            std::lock_guard<std::mutex> guard(fileCache_mutex);
            fileCache->erase(req.filename_start);
        } else {
            close(req.filename_start);
        }
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_LS) {
        // List files with filename
        uint32_t count = 0, offset = 0, entryCount = 0;
        for (const auto & entry : std::filesystem::directory_iterator(filename)) {
            entryCount++;
            if (count >= req.count) {
                continue;
            }
            if (offset >= req.offset) {
                auto path = entry.path();
                const char* s = path.c_str();
                int32_t len = strlen(s);
                if (count + 4 + len >= req.count) break;
                *(int32_t*)(toGPUBuf + req.result_start + count) = len;
                count += 4;
                memcpy(toGPUBuf + req.result_start + count, s, len);
                count += len;
            }
            offset++;
        }
        volatileReqs[i].offset = offset;
        volatileReqs[i].count = entryCount;
        volatileReqs[i].result_end = req.result_start + count;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_STAT) {
        // Stat a path and return file info
        struct stat st;
        int ok;
        if (req.offset == 1) {
            ok = lstat(filename, &st);
        } else {
            ok = stat(filename, &st);
        }
        if (verbose) fprintf(stderr, "Got %d from stat, req.count = %ld\n", ok, req.count);
        volatileReqs[i].result_end = ok == 0 ? req.result_start + 104 : 0;
        if (ok == 0 && req.count >= 104) {
            uint64_t *u64 = (uint64_t*)(toGPUBuf + req.result_start);
            int i = 0;
            u64[i++] = st.st_atim.tv_sec;
            u64[i++] = st.st_atim.tv_nsec;
            u64[i++] = st.st_mtim.tv_sec;
            u64[i++] = st.st_mtim.tv_nsec;
            u64[i++] = st.st_ctim.tv_sec;
            u64[i++] = st.st_ctim.tv_nsec;

            u64[i++] = st.st_ino;
            u64[i++] = st.st_size;
            u64[i++] = st.st_blocks;

            uint32_t *u32 = (uint32_t*)(u64 + i);
            i = 0;
            u32[i++] = st.st_dev;
            u32[i++] = st.st_mode;
            u32[i++] = st.st_nlink;
            u32[i++] = st.st_uid;
            u32[i++] = st.st_gid;
            u32[i++] = st.st_rdev;
            u32[i++] = st.st_blksize;
            ((int32_t*)(u32))[i++] = 0;

            if (verbose) fprintf(stderr, "stat copied: setting res_end %d\n", req.result_start + 104);
        } else if (ok == 0) {
        } else {
            *((int32_t*)(toGPUBuf + req.result_start + 100)) = errno;
            errno = 0;
        }
        if (verbose) fprintf(stderr, "stat res: ok %d errno %d res_end %d res_start %d\n", ok, *((int32_t*)(toGPUBuf + req.result_start + 100)), volatileReqs[i].result_end, req.result_start);
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_DLOPEN) {
        // Load shared library
        void *lib = dlopen(filename, RTLD_LAZY);
        *((int64_t*)(toGPUBuf + req.result_start)) = (int64_t)lib;
        volatileReqs[i].result_end = req.result_start + 8;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_DLCALL) {
        // Call a function in a shared library
        void *lib = (void*)req.offset;
        app->readFromGPUIO(req.result_start, req.count);
        void (*func)(void* src, uint32_t srcLength, void* dst, uint32_t dstLength);
        *(void **) (&func) = dlsym(lib, filename);
        (*func)((void*)(fromGPUBuf + req.result_start), int32_t(req.count), (void*)(toGPUBuf + req.data2_start), req.data2_end-req.data2_start);
        volatileReqs[i].result_end = 0;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_DLCLOSE) {
        // Close a shared library
        dlclose((void*)req.offset);
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_MALLOC) {
        // Allocate CPU memory
        void *ptr = malloc(req.count);
        *((int64_t*)(toGPUBuf + req.result_start)) = (int64_t)ptr;
        volatileReqs[i].result_end = req.result_start + 8;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_MEMWRITE) {
        // Write to CPU memory
        void *ptr = (void*)req.offset;
        memcpy(ptr, fromGPUBuf + req.result_start, req.count);
        volatileReqs[i].result_end = req.result_start + req.count;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_MEMREAD) {
        // Read from CPU memory
        void *ptr = (void*)req.offset;
        memcpy(toGPUBuf + req.result_start, ptr, req.count);
        volatileReqs[i].result_end = req.result_start + req.count;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_MEMFREE) {
        // Free CPU-side memory
        void *ptr = (void*)req.offset;
        free(ptr);
        volatileReqs[i].result_end = 0;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_LISTEN) {
        // Start listening for connections on TCP socket
        int32_t sockfd = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons( req.offset );
        bind(sockfd, (struct sockaddr*)&address, sizeof(address));
        listen(sockfd, 1024);
        if (verbose) fprintf(stderr, "IO_LISTEN: fd %d port %ld\n", sockfd, req.offset);
        volatileReqs[i].result_start = sockfd;
        volatileReqs[i].result_end = -2;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_ACCEPT) {
        // Accept connection on listening socket
        struct sockaddr_in address;
        socklen_t addrlen = sizeof(address);
        int32_t sockfd = accept(req.filename_start, (struct sockaddr*)&address, &addrlen);
        if (verbose) fprintf(stderr, "IO_ACCEPT: listening socket fd %d -> conn fd %d\n", req.filename_start, sockfd);
        if (req.count != 0) {
            int32_t bytes = recv(sockfd, toGPUBuf + req.data2_start, req.count, req.offset);
            volatileReqs[i].data2_end = req.data2_start + bytes;
//                fprintf(stderr, "Got data: %.4096s\n", toGPUBuf + req.data2_start);
        }
        volatileReqs[i].result_start = sockfd;
        volatileReqs[i].result_end = -2;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_CONNECT) {
        // Connect to a remote server - async (CPU starts creating connections, GPU goes on with its merry business, GPU writes to socket are pipelined on CPU side)
        req.status = IO_ERROR;

    } else if (req.ioType == IO_SEND) {
        // Send data to a socket
        app->readFromGPUIO(req.result_start, req.count);
        app->readFromGPUIO(req.data2_start, req.data2_end - req.data2_start);
        int32_t bytes = sendto(req.filename_start, fromGPUBuf + req.result_start, req.count, req.offset, (struct sockaddr*)(fromGPUBuf + req.data2_start), req.data2_end - req.data2_start);
        volatileReqs[i].result_end = req.result_start + bytes;
        if (req.progress != 0) {
            close(req.filename_start);
        }
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_RECV) {
        // Receive data from a socket
        socklen_t addrlen = req.data2_end - req.data2_start;
        int32_t bytes = recvfrom(req.filename_start, toGPUBuf + req.result_start, req.count, req.offset, (struct sockaddr*)(fromGPUBuf + req.data2_start), &addrlen);
        volatileReqs[i].data2_end = req.data2_start + addrlen;
        volatileReqs[i].result_end = req.result_start + bytes;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_CHROOT) {
        // Chdir to given dir and make it root
        int result = chdir(filename);
        if (result == 0) result = chroot(".");
        volatileReqs[i].result_start = result;
        volatileReqs[i].result_end = 0;
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_TIMENOW) {
        // Get current wallclock time in microseconds since the epoch.
        // For shader timings, you're probably better off with clockARB() and clockRealtimeEXT()
        int64_t time_us = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
        volatileReqs[i].offset = time_us;
        if (req.count >= 8) {
            *((int64_t*)(toGPUBuf + req.result_start)) = time_us;
            volatileReqs[i].result_end = req.result_start + req.result_end;
        } else {
            volatileReqs[i].result_end = 0;
        }
        req.status = IO_COMPLETE;

    } else if (req.ioType == IO_EXIT) {
        req.status = IO_COMPLETE;
        exit(req.offset); // Uh..

    } else {
        req.status = IO_ERROR;

    }
    volatileReqs[i].status = req.status;
    if (verbose) fprintf(stderr, "IO completed: %d - status %d\n", i, volatileReqs[i].status);

    if (threadIdx >= 0) {
        std::lock_guard<std::mutex> guard(completed_mutex);
        completed[threadIdx] = true;
    }
    signalIOProgress();
}

static void handleIORequests(ComputeApplication *app, bool verbose, ioRequests *ioReqs, char *toGPUBuf, char *fromGPUBuf, volatile bool *ioRunning, volatile bool *ioReset) {

    int threadCount = 64;
    std::thread threads[threadCount];
    int threadIdx = 0;
    volatile bool *completed = (volatile bool*)malloc(sizeof(bool) * threadCount);
    auto volatileReqs = ((volatile ioRequest*)(ioReqs->requests));

    std::map<int, FILE*> fileCache{};

    int32_t lastReqNum = 0;
    if (verbose) fprintf(stderr, "IO Running\n");
    while (*ioRunning) {
        if (*ioReset) {
            while (threadIdx > 0) threads[--threadIdx].join();
            for (int i=0; i < threadCount; i++) completed[i] = 0;
            lastReqNum = 0;
            ioReqs->ioCount = 0;
            *ioReset = false;
        }
        int32_t reqNum = ((volatile ioRequests*)ioReqs)->ioCount % IO_REQUEST_COUNT;
        if (reqNum < lastReqNum) {
            for (int32_t i = lastReqNum; i < IO_REQUEST_COUNT; i++) {
                if (verbose) fprintf(stderr, "Got IO request %d\n", i);
                int32_t ioType = volatileReqs[i].ioType;
                if (ioType == IO_ACCEPT) {
                    std::thread(handleIORequest, app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, -1, &fileCache).detach();
                } else if (ioType != IO_TIMENOW) {
                    int tidx = threadIdx;
                    if (tidx == threadCount) {
                        // Find completed thread.
                        for (int j = 0; j < threadCount; j++) {
                            if (completed[j]) {
                                tidx = j;
                                threads[j].join();
                                break;
                            }
                        }
                        if (tidx == threadCount) {
                            threads[0].join();
                            tidx = 0;
                        }
                    } else {
                        tidx = threadIdx++;
                    }
                    std::lock_guard<std::mutex> guard(completed_mutex);
                    completed[tidx] = false;
                    threads[tidx] = std::thread(handleIORequest, app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, tidx, &fileCache);
                } else {
                    handleIORequest(app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, -1, &fileCache);
                }
            }
            lastReqNum = 0;
        }
        for (int32_t i = lastReqNum; i < reqNum; i++) {
            if (verbose) fprintf(stderr, "Got IO request %d\n", i);
            if (volatileReqs[i].ioType == IO_ACCEPT) {
                std::thread(handleIORequest, app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, -1, &fileCache).detach();
            } else if (volatileReqs[i].ioType != IO_TIMENOW) {
                int tidx = threadIdx;
                if (tidx == threadCount) {
                    // Find completed thread.
                    for (int j = 0; j < threadCount; j++) {
                        if (completed[j]) {
                            tidx = j;
                            threads[j].join();
                            break;
                        }
                    }
                    if (tidx == threadCount) {
                        threads[0].join();
                        tidx = 0;
                    }
                } else {
                    tidx = threadIdx++;
                }
                std::lock_guard<std::mutex> guard(completed_mutex);
                completed[tidx] = false;
                threads[tidx] = std::thread(handleIORequest, app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, tidx, &fileCache);
            } else {
                handleIORequest(app, verbose, ioReqs, toGPUBuf, fromGPUBuf, i, completed, -1, &fileCache);
            }

        }
        lastReqNum = reqNum;
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    while (threadIdx > 0) threads[--threadIdx].join();
    for (int i=0; i < threadCount; i++) completed[i] = false;
    if (verbose) fprintf(stderr, "Exited IO thread\n");
}

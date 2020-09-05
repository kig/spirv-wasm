io dlcall(uint64_t lib, string symbol, alloc_t args, alloc_t result) {
    return requestIO(ioRequest(IO_DLCALL, IO_START, int64_t(lib), strLen(args), symbol, args, 0,0,result,0,0));
}

io dlopen(string path, alloc_t dstBuffer) {
    return requestIO(ioRequest(IO_DLOPEN, IO_START, 0, 0, path, dstBuffer, 0,0,string(0,0),0,0));
}

uint64_t dlopenSync(string path) {
    uint64_t lib;
    FREE(FREE_IO(
        string res = awaitIO(dlopen(path, malloc(8)), true);
        lib = readU64fromIO(res.x);
    ))
    return lib;
}

string dlcallSync(uint64_t lib, string symbol, alloc_t args, alloc_t result) {
    string res;
    FREE_IO( res = awaitIO(dlcall(lib, symbol, args, result)); )
    return res;
}

void dlcallSync(uint64_t lib, string symbol, alloc_t args) {
    dlcallSync(lib, symbol, args, string(-4,-4));
}

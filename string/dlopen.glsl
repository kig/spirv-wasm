uint64_t dlopenSync(string path) {
    uint64_t lib;
    FREE(FREE_IO(
        io r = requestIO(ioRequest(IO_DLOPEN, IO_START, 0, 0, path, malloc(8), 0,0,string(0,0),0,0));
        string res = awaitIO(r, true);
        lib = readU64fromIO(res.x);
    ))
    return lib;
}

string dlcallSync(uint64_t lib, string symbol, string args, int32_t count) {
    string res;
    FREE_IO(
        io r = requestIO(ioRequest(IO_DLCALL, IO_START, int64_t(lib), strLen(args), symbol, args, 0,0,malloc(count),0,0));
        res = awaitIO(r);
    )
    return res;
}

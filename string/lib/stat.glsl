struct Nanotime {
    uint64_t tv_sec;
    uint64_t tv_nsec;
};
const int32_t NanotimeSize = 16;

struct Stat {
    Nanotime st_atim;
    Nanotime st_mtim;
    Nanotime st_ctim;

    uint64_t st_ino;
    uint64_t st_size;
    uint64_t st_blocks;

    uint32_t st_dev;
    uint32_t st_mode;
    uint32_t st_nlink;
    uint32_t st_uid;
    uint32_t st_gid;
    uint32_t st_rdev;
    uint32_t st_blksize;

    int32_t error;
};
const int32_t StatSize = 3 * NanotimeSize + 3 * 8 + 8 * 4; // 104

Stat initStat(string s) {
    Stat st;
    if (strLen(s) < StatSize) {
        st.error = -1;
        return st;
    }
    ptr_t i = s.x;

    st.st_atim.tv_sec = readU64heap(i); i+=8;
    st.st_atim.tv_nsec = readU64heap(i); i+=8;
    st.st_mtim.tv_sec = readU64heap(i); i+=8;
    st.st_mtim.tv_nsec = readU64heap(i); i+=8;
    st.st_ctim.tv_sec = readU64heap(i); i+=8;
    st.st_ctim.tv_nsec = readU64heap(i); i+=8;

    st.st_ino = readU64heap(i); i+=8;
    st.st_size = readU64heap(i); i+=8;
    st.st_blocks = readU64heap(i); i+=8;

    st.st_dev = readU32heap(i); i+=4;
    st.st_mode = readU32heap(i); i+=4;
    st.st_nlink = readU32heap(i); i+=4;
    st.st_uid = readU32heap(i); i+=4;
    st.st_gid = readU32heap(i); i+=4;
    st.st_rdev = readU32heap(i); i+=4;
    st.st_blksize = readU32heap(i); i+=4;

    st.error  = readI32heap(i); i+=4;
    return st;
}

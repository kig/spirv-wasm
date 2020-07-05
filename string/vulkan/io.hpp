#define IO_NONE 0
#define IO_START 1
#define IO_RECEIVED 2
#define IO_IN_PROGRESS 3
#define IO_COMPLETE 4
#define IO_ERROR 5
#define IO_HANDLED 255



#define IO_READ 1
#define IO_WRITE 2
#define IO_CREATE 3
#define IO_DELETE 4
#define IO_TRUNCATE 5

#define IO_LISTEN 6
#define IO_ACCEPT 7
#define IO_CLOSE 8
#define IO_OPEN 9
#define IO_FSYNC 10
#define IO_SEND 11
#define IO_RECV 12
#define IO_TIMENOW 13
#define IO_TIMEOUT 14
#define IO_CD 15
#define IO_LS 16
#define IO_RMDIR 17
#define IO_CONNECT 24
#define IO_GETCWD 25
#define IO_STAT 26
#define IO_MOVE 27
#define IO_COPY 28
#define IO_COPY_RANGE 29
#define IO_MKDIR 30

#define IO_RUN_CMD 31
#define IO_EXIT 32

#define IO_POPEN 33
#define IO_PCLOSE 34

// IO system benchmarking functions
// How fast can you handle IO requests?
#define IO_NOP 250
// How fast can you send data to CPU and get it back?
#define IO_PINGPONG 251


/*
The design for GPU IO
---

Workgroups submit tasks in sync -> readv / writev approach is beneficial for sequential reads/writes. Readv/writev are internally single-threaded, so limited by memcpy to 6-8 GB/s.

Lots of parallelism.
Ordering of writes across workgroups requires a way to sequence IOs (either reduce to order on the GPU or reassemble correct order on the CPU.)

Compression of data on the PCIe bus would help. 32 * zstd --format=lz4 --fast -T1 file -o /dev/null goes at 38 GB/s.

Need benchmark suite
---

 - Different block sizes
 - Different access patterns (sequential, random)
     - Scatter writes
     - Sequential writes
     - Gather reads
     - Sequential reads
     - Combined reads & writes
 - Different levels of parallelism
    - 1 IO per thread group
    - each thread does its own IO
    - 1 IO on ThreadID 0
    - IOs across all invocation
 - Compression
 - From hot cache on CPU
 - From cold cache
 - With GPU-side cache
 - Repeated access to same file
 - Access to multiple files

Does it help to combine reads & writes into sequential blocks on CPU-side when possible, or is it faster to do IOs ASAP?

Caching file descriptors, helps or not?



IORING_OP_NOP
// This operation does nothing at all; the benefits of doing nothing asynchronously are minimal, but sometimes a placeholder is useful.
IORING_OP_READV
IORING_OP_WRITEV
// Submit a readv() or write() operation — the core purpose for io_uring in most settings.
IORING_OP_READ_FIXED
IORING_OP_WRITE_FIXED
// These opcodes also submit I/O operations, but they use "registered" buffers that are already mapped into the kernel, reducing the amount of total overhead.
IORING_OP_FSYNC
// Issue an fsync() call — asynchronous synchronization, in other words.
IORING_OP_POLL_ADD
IORING_OP_POLL_REMOVE
// IORING_OP_POLL_ADD will perform a poll() operation on a set of file descriptors. It's a one-shot operation that must be resubmitted after it completes; it can be explicitly canceled with IORING_OP_POLL_REMOVE. Polling this way can be used to asynchronously keep an eye on a set of file descriptors. The io_uring subsystem also supports a concept of dependencies between operations; a poll could be used to hold off on issuing another operation until the underlying file descriptor is ready for it.

IORING_OP_SYNC_FILE_RANGE
// Perform a sync_file_range() call — essentially an enhancement of the existing fsync() support, though without all of the guarantees of fsync().
IORING_OP_SENDMSG
IORING_OP_RECVMSG (5.3)
// These operations support the asynchronous sending and receiving of packets over the network with sendmsg() and recvmsg().
IORING_OP_TIMEOUT
IORING_OP_TIMEOUT_REMOVE
// This operation completes after a given period of time, as measured either in seconds or number of completed io_uring operations. It is a way of forcing a waiting application to wake up even if it would otherwise continue sleeping for more completions.
IORING_OP_ACCEPT
IORING_OP_CONNECT
// Accept a connection on a socket, or initiate a connection to a remote peer.
IORING_OP_ASYNC_CANCEL
// Attempt to cancel an operation that is currently in flight. Whether this attempt will succeed depends on the type of operation and how far along it is.
IORING_OP_LINK_TIMEOUT
// Create a timeout linked to a specific operation in the ring. Should that operation still be outstanding when the timeout happens, the kernel will attempt to cancel the operation. If, instead, the operation completes first, the timeout will be canceled.

IORING_OP_FALLOCATE
// Manipulate the blocks allocated for a file using fallocate()
IORING_OP_OPENAT
IORING_OP_OPENAT2
IORING_OP_CLOSE
// Open and close files
IORING_OP_FILES_UPDATE
// Frequently used files can be registered with io_uring for faster access; this command is a way of (asynchronously) adding files to the list (or removing them from the list).
IORING_OP_STATX
// Query information about a file using statx().
IORING_OP_READ
IORING_OP_WRITE
// These are like IORING_OP_READV and IORING_OP_WRITEV, but they use the simpler interface that can only handle a single buffer.
IORING_OP_FADVISE
IORING_OP_MADVISE
// Perform the posix_fadvise() and madvise() system calls asynchronously.
IORING_OP_SEND
IORING_OP_RECV
// Send and receive network data.
IORING_OP_EPOLL_CTL
// Perform operations on epoll file-descriptor sets with epoll_ctl()

*/

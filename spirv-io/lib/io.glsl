// IO status codes

#define IO_NONE 0
#define IO_START 1
#define IO_RECEIVED 2
#define IO_IN_PROGRESS 3
#define IO_COMPLETE 4
#define IO_ERROR 5
#define IO_HANDLED 255

// Program repetition

#define NO_RERUN 0
#define RERUN_NOW 1
#define RERUN_ON_IO 2

// IO compression

#define IO_COMPRESS_SPEED_0 0x00100000
#define IO_COMPRESS_SPEED_1 0x00200000
#define IO_COMPRESS_SPEED_2 0x00300000
#define IO_COMPRESS_SPEED_3 0x00400000
#define IO_COMPRESS_SPEED_4 0x00500000
#define IO_COMPRESS_SPEED_5 0x00600000
#define IO_COMPRESS_SPEED_6 0x00700000
#define IO_COMPRESS_SPEED_7 0x00800000
#define IO_COMPRESS_SPEED_8 0x00900000
#define IO_COMPRESS_SPEED_9 0x00a00000

// LZ4 compressed stream of independent blocks with block size in the lower 24 bits
#define IO_COMPRESS_LZ4_BLOCK_STREAM (3<<24)
// Single LZ4 compressed block
#define IO_COMPRESS_LZ4_BLOCK (2<<24)
// LZ4 frame stream (Found in e.g. .lz4 files)
#define IO_COMPRESS_LZ4 (1<<24)

// ZSTD frame stream - No GPU decompressor
#define IO_COMPRESS_ZSTD (4<<24)


// IO calls

// Read from a file at given offset.
#define IO_READ 1
// Read a line from a file pointer at current offset.
#define IO_READLINE 41
// Write to a file at given offset.
#define IO_WRITE 2
// Create new file, optionally write initial data to it.
#define IO_CREATE 3
// Delete a file.
#define IO_DELETE 4
// Change the length of file.
#define IO_TRUNCATE 5

// Get working directory.
#define IO_GETCWD 25
// Change working directory.
#define IO_CD 15
// List files in a directory. Should be extended to pass stat data.
#define IO_LS 16
// Delete a directory.
#define IO_RMDIR 17

// Stat a file - passes raw stat struct, needs GLSL lib changes
#define IO_STAT 26

// Move a file
#define IO_MOVE 27
// Copy a file
#define IO_COPY 28
// Create a directory.
#define IO_MKDIR 30

// Do a system() call.
#define IO_RUN_CMD 31

// exit() the process.
#define IO_EXIT 32

// Load a shared library into the IO runtime
#define IO_DLOPEN 38
// Call a function in a loaded shared library
#define IO_DLCALL 39
// Close a shared library object and release its memory
#define IO_DLCLOSE 40

// Allocate memory on the CPU
#define IO_MALLOC 42
// Read from CPU memory
#define IO_MEMREAD 43
// Write to CPU memory
#define IO_MEMWRITE 44
// Free allocated CPU memory
#define IO_MEMFREE 45

// IO system benchmarking functions
// How fast can you handle IO requests?
#define IO_NOP 250
// How fast can you send data to CPU and get it back?
#define IO_PINGPONG 251



// Listen for connections on a TCP socket
#define IO_LISTEN 6
// Accept a new connection from a listening socket
#define IO_ACCEPT 7
// Close an open file
#define IO_CLOSE 8
// Open a file and return the file id
#define IO_OPEN 9

// fsync an open file - NOT IMPLEMENTED
#define IO_FSYNC 10

// Send data to a socket
#define IO_SEND 11
// Received data from a socket
#define IO_RECV 12

// Get current time.
#define IO_TIMENOW 13
// Wait for a duration on the CPU side. - NOT IMPLEMENTED
#define IO_TIMEOUT 14

// Connect to a remote server - NOT IMPLEMENTED
#define IO_CONNECT 24

// Copy a range of bytes from one file to another - NOT IMPLEMENTED
#define IO_COPY_RANGE 29

// Open a pipe to a process - NOT IMPLEMENTED
#define IO_POPEN 33
// Close process pipe. - NOT IMPLEMENTED
#define IO_PCLOSE 34

// Invoke other compute kernels (TBD how this works) - NOT IMPLEMENTED
#define IO_RUNSPV 35

// Chrooting the IO process might be useful
#define IO_CHROOT 36

// In case you want to stop talking to the kernel - NOT IMPLEMENTED
#define IO_SECCOMP 37


/*
Design of GPU IO
===

Hardware considerations
---

 - GPUs have around a hundred processors, each with a 32-wide SIMD unit.
 - The SIMD unit can execute a 32 thread threadgroup and juggle between ten threadgroups for latency hiding.
 - GPU cacheline is 128 bytes.
 - CPU cacheline is 64 bytes.
 - GPU memory bandwidth is 400 - 1000 GB/s.
 - CPU memory bandwidth is around 50 GB/s.
 - PCIe bandwidth 11-13 GB/s. On PCIe4, 20 GB/s.
 - NVMe flash can do 2.5-10 GB/s on 4-16 channels. PCIe4 could boost to 5-20 GB/s.
 - The CPU can do 30 GB/s memcpy with multiple threads, so it's possible to keep PCIe4 saturated even with x16 -> x16.
 - GPUdirect access to other PCIe devices is only available on server GPUs. Other GPUs need a roundtrip via CPU.
 - CPU memory accesses require several threads of execution to hit full memory bandwidth (single thread can do ~15 GB/s)
 - DRAM is good at random access at >cacheline chunks with ~3-4x the bandwidth of PCIe3 x16, ~2x PCIe4 x16.
 - Flash SSDs are good at random access at >128kB chunks, perform best with sequential accesses, can deal with high amounts of parallel requests. Writes are converted to log format.
 - Optane is good at random access at small sizes >4kB and low parallelism. The performance of random and sequential accesses is similar.

 * => Large reads to flash should be executed in sequence (could be done by prefetching the entire file to page cache and only serving requests once the prefetcher has passed them)
 * => Small scattered reads should be dispatched in parallel (if IO rate < prefetch speed, just prefetch the whole file)
 * => Writes can be dispatched in parallel with more freedom, especially without fsync. Sequential and/or large block size writes will perform better on flash.
 * => Doing 128 small IO requests in parallel may perform better than 16 parallel requests.
 * => IOs to page cache should be done in parallel and ASAP.
 * => Caching data into GPU RAM is important for performance.
 * => Programs that execute faster than the PCIe bus should be run on the CPU if the GPU doesn't have the data in cache.
 * => Fujitsu A64FX-type designs with lots of CPU cores with wide vector units and high bandwidth memory are awesome. No data juggling, no execution environment weirdness.

Software
---

The IO queue works by using spinlocks on both the CPU and GPU sides.
The fewer IO requests you make, the less time you spend spinning.
Sending data between CPU and GPU works best in large chunks.

To avoid issues with cacheline clashes, align messages on GPU cacheline size.
IO request spinlocks that read across the PCIe bus should have small delays between checks to avoid hogging the PCIe bus.
Workgroups (especially subgroups) should bundle their IOs into a single scatter/gather.

When working with opened files, reads and writes should be done with pread/pwrite. Sharing a FILE* across threads isn't a great idea.
The cost of opening and closing files with every IO is eclipsed by transfer speeds with large (1 MB) block sizes.

The IO library should be designed for big instructions with minimal roundtrips.
E.g. directory listings should send the entire file list with file stats, and there should be a recursive version to transfer entire hierarchies.
Think more shell utilities than syscalls. Use CPU as IO processor that can do local data processing without involving the GPU.

Workgroup concurrency can be used to run the same code on CPU and GPU in parallel. This extends to multi-GPU and multi-node quite naturally.
The IO queue could be used to exchange data between running workgroups.

Limited amount of memory that can be shared between CPU and GPU (I start seeing issues with > 64 MB allocations).
Having a small IO heap for each thread or even threadgroup, while easy to parallelize, limits IO sizes severely.
32 MB transfer buffer, 32k threads -> 1k max IO per thread, or 32k per 32-wide subgroup.
Preferable to do 1+ MB IOs.
Design with a concurrently running IO manager program that processes IO transfers?
The CPU could also manage this by issuing copyBuffer calls to move data.

Workgroups submit tasks in sync -> readv / writev approach is beneficial for sequential reads/writes.
Readv/writev are internally single-threaded, so probably limited by memcpy to 6-8 GB/s.

Ordering of writes across workgroups requires a way to sequence IOs (either reduce to order on the GPU or reassemble correct order on the CPU.)
IOs could have sequence ids.

Compression of data on the PCIe bus could help. 32 * zstd --format=lz4 --fast -T1 file -o /dev/null goes at 38 GB/s.

Caching file data on the GPU is important for performance, 40x higher bandwidth than CPU page cache over PCIe.
Without GPU-side caching, you'll likely get better perf on the CPU on bandwidth-limited tasks (>50 GB/s throughput.)
In those tasks, using memory bandwidth to send data to GPU wouldn't help any, best you could achieve is zero slowdown.
(Memory bandwidth 50 GB/s. CPU processing speed 50 GB/s. Use 10 GB/s of bandwidth to send data to GPU =>
CPU has only 40 GB/s bandwidth left, GPU can do 10 GB/s => CPU+GPU processing speed 50 GB/s.)

The IO layer should do the "right thing", e.g. scatter writes to a single filename should all go to the same fd using pwrite. The IO layer
should also try to turn scattered reads and writes into sequential ones if they're not to DRAM / Optane. Mixed read-write workloads into the same
file should have the reads happening from the same fd as the writes are going to.

For some tasks it'd be nice to command the IO layer without involving the GPU. Maybe send a small shader program to run on the IO processor (ha ha).
For example, the IO runtime can compress LZ4 frames at 22 GB/s. This is useful by itself: write it to a file, write it over network, etc.
No need to involve the GPU in that.

But how to do this sort of DMA IO redirect? It is like the copyrange / splice stuff. You have two fds, then tell the kernel (or IO runtime in this case)
to read from one and write into the other. Except that now you also tell it to do some stuff like "pull these chunks in parallel from this file,
compress them in parallel, and put them into this other file - in order" (The last part is important.)

You'd issue parallel compressed reads but tell the IO runtime to toss the data into a CPU-side memory buffer. Then you'd do a prefix sum over the reads in order
to calculate write offsets. As soon as you'd have the offset for a write, you'd issue a write to move the data from the CPU memory buffer to the target file.
The reason for issuing the write ASAP is, well, sequential memcpy peaks at 7 GB/s. If your compressor is doing 22 GB/s, need 4 parallel writes.

For piping, you'll be stuck to single memcpy speeds, and would issue writes after previous writes have finished. Don't know about networking, how do you hit
those 200-400 Gbps rates.

Custom IO handlers coupled with shared library loading would make it possible to use custom CPU-side functionality from the shaders.
For example, say you wanted to stream a fractal animation as MJPEG. The shader would compute a frame in a fractal animation,
then use libturbojpeg on the CPU to compress it, keeping the result on the CPU memory. Finally, the shader would ask the IO layer
to write the result buffer into the network socket.

For this example, you'd need shared library loading (got it), keeping the result on the CPU side, and telling the CPU to do the write
using the CPU side buffer. Or you could register a IO_WRITE_MPJPEG handler that'd receive {uint32_t width; uint32_t height; uint8_t data;},
convert the image into JPEG, write the header and the image into the socket.

How about doing a GUI application? You'd have to expose a window frame buffer to the compositor and receive input events.
How about a compositor? You'd have to receive window frame buffers that you'd then composite onto a display surface and flip to the screen.


Benchmark suite
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



Running multiple programs on the same compute application instance
---

Running multiple programs can provide benefits in the following ways:

 - Lower startup latency (< 1 ms for cached pipelines if you don't need to allocate buffers etc.)
     - Important for GPU, where Vulkan startup latency is in the hundreds of milliseconds range.
     - Less important on the CPU.
 - Share buffers between programs
     - No need to move data in and out of GPU RAM
     - Shared page cache for file I/O
     - Can write application as several co-operating programs vs. one megaprogram
 - Fully use GPU when each program can use only a part of the GPU
 - Better latency hiding for I/O (program waiting for I/O -> run another program)
 - Share CPU memory between multi-GPU programs


Multi-GPU support
---

Allocate memory on other GPUs and copy data over using fast p2p bus.
Run different programs on different GPUs for e.g. pipeline processing.
Sync with other GPUs.
Treat CPU NUMA nodes as distinct "GPUs".
Networked programs using the same strategy.




Appendix A: io_uring operations
---

IO_uring's design feels a bit low-level for GPU ops

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

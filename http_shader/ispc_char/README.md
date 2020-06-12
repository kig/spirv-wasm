Change the buffer type from int to int8 in the generated ISPC files.

Easiest way to run this is without rebuilding the ISPC files.

```bash
ispc -O3 -o httpd.ispc.o runner.ispc &&
clang++ -I/usr/local/bin -pthread -std=c++11 -lm -pthread -O3 -o httpd httpd.ispc.o ../tasksys.cpp httpd_ispc.cpp &&
sh run.sh
# 59
# HTTP/1.1 200 OK
# Content-Type: text/plain
#
# Hello, World!
# Elapsed: 25451 ms
# Million requests per second: 10.300
# 423.24user 2.08system 0:25.56elapsed 1663%CPU (0avgtext+0avgdata 790060maxresident)k
# 0inputs+0outputs (0major+262349minor)pagefaults 0swaps
```

If you want to do a full build:

```bash
(cd .. && sh build_ispc_char.sh)
# Do the type change in httpd.ispc, changing int to int8 in the buffers.
sh run.sh
```

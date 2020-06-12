Change the request and response buffer types from int to int8 in the generated ISPC files for ~2x perf.

Easiest way to run this is without rebuilding the ISPC files.

```bash
ispc -O3 -o httpd.ispc.o runner.ispc &&
clang++ -I/usr/local/bin -pthread -std=c++11 -lm -pthread -O3 -o httpd httpd.ispc.o ../tasksys.cpp httpd_ispc.cpp &&
sh run.sh
# 367
# HTTP/1.1 200 OK
# Content-Type: text/plain
#
# Hello, World!
# Host: localhost:9000
# User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:76.0) Gecko/20100101 Firefox/76.0
# Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
# Accept-Language: en-US,en;q=0.5
# Accept-Encoding: gzip, deflate
# Connection: keep-alive
# Upgrade-Insecure-Requests:
#
# Elapsed: 2848 ms
# Million requests per second: 11.506
#
# 84.56user 0.44system 0:02.86elapsed 2965%CPU (0avgtext+0avgdata 98164maxresident)k
# 0inputs+0outputs (0major+23821minor)pagefaults 0swaps
```

If you want to do a full build:

```bash
(cd .. && sh build_ispc_int.sh)
# Do the type change in httpd.ispc, changing int to int8 in the request and response buffers.
sh run.sh
```

Change the buffer type from int to int8 in the generated ISPC files.

Easiest way to run this is without rebuilding the ISPC files.

```
ispc -O3 -o httpd.ispc.o runner.ispc &&
clang++ -I/usr/local/bin -pthread -std=c++11 -lm -pthread -O3 -o httpd httpd.ispc.o ../tasksys.cpp httpd_ispc.cpp &&
sh run.sh
```

If you want to do a full build:

```
(cd .. && sh build_ispc_char.sh)
# Do the type change in httpd.ispc, changing int to int8 in the buffers.
sh run.sh
```

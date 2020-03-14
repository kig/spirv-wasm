# Dockerfile for ISPC to WASM compilation

First build the image: `docker build -t ispc-wasm:latest .`

Then you can compile ISPC files to WASM:
```bash
$ ./ispc2wasm.sh mandelbrot.ispc
$ ls
mandelbrot.ispc mandelbrot.ispc.o mandelbrot.ispc.wasm
```

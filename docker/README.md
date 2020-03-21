# Dockerfile for ISPC to WASM and GLSL to WASM compilation

First build the image: `docker build -t ispc-wasm:latest .`

Then you can compile ISPC files to WASM:

```bash
$ ./ispc2wasm.sh mandelbrot.ispc
$ ls
mandelbrot.ispc mandelbrot.ispc.o mandelbrot.ispc.wasm
```

And GLSL compute shaders to WASM (this is even more experimental!):

```bash
$ ./glsl2wasm.sh ao.comp.glsl
$ ls
ao.comp.glsl.html ao.comp.glsl.js ao.comp.glsl.worker.js ao.comp.glsl.wasm
```

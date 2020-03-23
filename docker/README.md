# Dockerfile for ISPC to WASM and GLSL to WASM compilation

First build the image: `docker build -t ispc-wasm:latest .`

## ISPC

Then you can compile ISPC files to WASM:

```bash
$ ./ispc2wasm.sh mandelbrot.ispc
$ ls
mandelbrot.ispc mandelbrot.ispc.o mandelbrot.ispc.wasm
```

To use from JavaScript:

```js
// Load the ISPC module
const obj = await WebAssembly.instantiateStreaming(fetch('mandelbrot.ispc.wasm'), {"env": {
    "ISPCAlloc":() => console.log("ISPCAlloc"),
    "ISPCLaunch":() => console.log("ISPCLaunch"), 
    "ISPCSync":() => console.log("ISPCSync"), 
}});
const width = 1920, height = 1080, outputPtr = 0;
obj.instance.exports.memory.grow(Math.ceil(width * height * 4 / 2**16)); // Allocate space for output image
instance.exports.mandelbrot_ispc(-2.5, -1, 1, 1, width, height, 255, outputPtr); // Call the ISPC function

const heap = new Int32Array(obj.instance.exports.memory.buffer); // Read in the output image off the heap.

const canvas = document.createElement('canvas');
canvas.width = width;
canvas.height = height;
const ctx = canvas.getContext('2d');
const id = ctx.createImageData(width, height);

for (let i = 0; i < width*height; i++) {
    id.data[i*4 + 0] = heap[i];
    id.data[i*4 + 1] = heap[i];
    id.data[i*4 + 2] = heap[i];
    id.data[i*4 + 3] = 255;
}

ctx.putImageData(id, 0, 0);
document.body.append(canvas);
```

## GLSL

Compile GLSL compute shaders to WebAssembly (this is even more experimental!):

```bash
$ ./glsl2wasm.sh ao.comp.glsl
$ ls
ao.comp.glsl.html ao.comp.glsl.js ao.comp.glsl.worker.js ao.comp.glsl.wasm
```

To use the GLSL version, open `ao.comp.glsl.html` and run:

```js
const width = 1920, height = 1080;
const localSizeX = 192, localSizeY = 10; // Local workgroup size of the compute shader.
// Spawn enough workgroups to cover the image.
const numWorkGroupsX = width / localSizeX;
const numWorkGroupsY = height / localSizeY;
const numWorkGroupsZ =  1;

const inputPtr = Module._malloc(8*4); // The shader takes an 8-float SSBO as its input buffer.
const outputPtr = Module._malloc(width*height*4); // And writes to a 8-bit RGBA image buffer.

const input = new Float32Array(Module.wasmMemory.buffer, inputPtr, 8);
input.set([width, height, 0, 0, 0, 0, 0, 0]); // Write the SSBO values to the input buffer.

// Run the shader across all accessible cores and SIMD lanes.
Module._run(numWorkGroupsX, numWorkGroupsY, numWorkGroupsZ, inputPtr, outputPtr);

// Show the resulting image.
const canvas = document.createElement('canvas');
canvas.width = width;
canvas.height = height;
const ctx = canvas.getContext('2d');
const id = ctx.createImageData(width, height);
id.data.set(new Uint8Array(Module.wasmMemory.buffer, outputPtr, id.data.byteLength));
ctx.putImageData(id, 0, 0);
document.body.append(canvas);
```
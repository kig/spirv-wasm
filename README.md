# spirv-wasm

Run SPIR-V shaders in WebAssembly

See demo at https://fhtr.org/spirv-wasm - uses WebAssembly Threads, so you may need to turn them on in chrome://flags or such.

## Project goal

Write low-effort code in one language, run safely in the browser at 50% of hardware peak performance. (FWIW naive C++ achieves 1% of HW peak performance on a 16-core. If you include the GPU, it's closer to 0.1% of peak perf. Divide by amount of compute nodes available to you.)

## Build

Requires Emscripten, glslangValidator, glm and spirv-cross.

[Install Emscripten](https://emscripten.org/docs/getting_started/downloads.html)

The others are likely in your package manager.

```bash
brew install glslangValidator
brew install spirv-cross
brew install glm
```

Now you can build the shader:

```bash
source somewhere/emsdk/emsdk_env.sh
emmake make
serve
```

If everything went right, you can open [http://localhost:5000/src/mandel.html](http://localhost:5000/src/mandel.html)
and hopefully see a Mandelbrot fractal. Check the browser console for timings.


## Debug information

You can make Emscripten emit source maps with the -g4 flag. You can make glslangValidator and the latest versions of spirv-cross emit line numbers like this:

```
glslangValidator -g -V -o mandel.spv mandel.comp
spirv-cross --cpp --emit-line-directives --output mandel.spv.cpp mandel.spv
```

Then you just have to hack the Emscripten source maps to map from the cpp to the compute shader line numbers.

Send a PR if you do!


FROM ispc/ubuntu_16.04:llvm100

RUN apt update
RUN apt install -y libxml2 emacs-nox

WORKDIR /usr/local/src/wasm-ld
RUN wget -qO- https://storage.googleapis.com/webassembly/emscripten-releases-builds/linux/65d33d604d3fa0ebe03548378b898fc6608e9cb8/wasm-binaries.tbz2 | tar xj

WORKDIR /usr/local/src/ispc
RUN git pull
RUN mkdir build; cd build && cmake .. -DWASM_ENABLED=ON && make -j32
ENV PATH="/usr/local/src/ispc/build/bin:/usr/local/src/wasm-ld/install/bin/:${PATH}"

WORKDIR /usr/local/src/spirv

RUN apt install -y libglm-dev

RUN git clone --depth=1 https://github.com/GameTechDev/SPIRV-Cross SPIRV-Cross-ISPC
RUN git clone --depth=1 https://github.com/KhronosGroup/SPIRV-Cross
RUN git clone --depth=1 https://github.com/KhronosGroup/glslang

ENV CXX="clang++"
ENV CC="clang"

WORKDIR /usr/local/src/spirv/glslang
RUN mkdir build; cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j32 install

WORKDIR /usr/local/src/spirv/SPIRV-Cross
RUN make -j32 && cp spirv-cross /usr/bin

WORKDIR /usr/local/src/spirv/SPIRV-Cross-ISPC
COPY spirv-cross-linux-x86-64 /usr/bin/spirv-cross-ispc

WORKDIR /usr/local/src
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN git clone --depth=1 https://github.com/aschrein/ispc/ ispc-wasm

WORKDIR /usr/local/src/ispc-wasm
RUN git pull origin wasm_tests:wasm_tests && git checkout wasm_tests
RUN mkdir build && cd build && cmake .. -DWASM_ENABLED=ON && make -j32 ispc
ENV PATH="/usr/local/src/ispc-wasm/build/bin:/usr/local/src/wasm-ld/install/bin/:${PATH}"

WORKDIR /usr/local/src
RUN git clone https://github.com/emscripten-core/emsdk && cd emsdk && ./emsdk install latest && ./emsdk activate latest
ENV PATH="/usr/local/src/emsdk:/usr/local/src/emsdk/node/12.9.1_64bit/bin:/usr/local/src/emsdk/upstream/emscripten:${PATH}"

WORKDIR /usr/local/src/spirv-wasm
RUN apt install -y ruby
RUN echo "int main(int argc, char *argv[]) { return 0; }" > test.cpp && emcc test.cpp -o TEST.html -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=16 -s TOTAL_MEMORY=268435456 -msimd128 -s SIMD=1
RUN printf "#include <iostream>\nusing namespace std;\nint main() { cout << \"foo\"; return 0; }" > test.cpp && em++ -lpthread -fno-exceptions test.cpp -O3 -o TEST.html -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=16 -s TOTAL_MEMORY=268435456 -msimd128 -s SIMD=1 -s EXTRA_EXPORTED_RUNTIME_METHODS='["ccall"]'
COPY ./spirv-runner .
RUN make tasksys.o builtins.o 
RUN make ispcRunner.o

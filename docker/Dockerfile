FROM ispc/ubuntu_16.04:llvm100

RUN apt update
RUN apt install -y libxml2 emacs-nox

WORKDIR /usr/local/src/wasm-ld
RUN wget -qO- https://storage.googleapis.com/webassembly/emscripten-releases-builds/linux/65d33d604d3fa0ebe03548378b898fc6608e9cb8/wasm-binaries.tbz2 | tar xj

WORKDIR /usr/local/src/ispc
RUN git pull
RUN mkdir build; cd build && cmake .. -DWASM_ENABLED=ON && make -j32
ENV PATH="/usr/local/src/ispc/build/bin:/usr/local/src/wasm-ld/install/bin/:${PATH}"


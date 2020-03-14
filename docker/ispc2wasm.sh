#!/bin/bash

if [[ -z $1 ]]
then
 echo
 echo USAGE $0 source.ispc
 echo Generates source.ispc.o and source.ispc.wasm
 echo
 exit 1
fi

docker run -i -v `pwd`:/tmp --rm ispc-wasm:latest bash -c "ispc --target=wasm-i32x4 --nostdlib --emit-llvm-text -o - /tmp/$1 | llc -O3 -filetype=obj - -o /tmp/$1.o && wasm-ld --no-entry --export-all --allow-undefined -o /tmp/$1.wasm /tmp/$1.o"


#!/bin/bash

if [[ -z $1 ]]
then
 echo
 echo USAGE $0 source.glsl
 echo Generates source.glsl.html, source.glsl.worker.js, source.glsl.js, and source.glsl.wasm
 echo
 exit 1
fi

docker run -i -v `pwd`:/tmp --rm ispc-wasm:latest bash -c "cd /usr/local/src/spirv-wasm && cp /tmp/$1 program.comp.glsl && make TARGET="$1" build && cp $1.{html,wasm,worker.js,js} /tmp/"


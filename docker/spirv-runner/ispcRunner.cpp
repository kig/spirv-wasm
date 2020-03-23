#include <stdio.h>
#include <stdlib.h>
#include "program.h"
#include "emscripten.h"

#ifdef WIN32
#include <io.h>
#include <fcntl.h>
#endif

static uint32_t bufferSize = 0;
static uint32_t inputBufferSize = 0;
static uint32_t vulkanDeviceIndex = 0;
static int32_t workSize[3] = {1, 1, 1};

static char *input;

void readHeader()
{
    ::size_t input_length = 0, read_bytes = 0, input_buffer_size = 4096;

#ifdef WIN32
	_setmode(_fileno(stdout), _O_BINARY);
	_setmode(_fileno(stdin), _O_BINARY);
#endif

    bufferSize = 0;
    read_bytes = fread(&bufferSize, 1, 4, stdin);
    if (read_bytes < 4)
    {
        fprintf(stderr, "read only %zd bytes, using default bufferSize\n", read_bytes);
        bufferSize = 4;
    }

    vulkanDeviceIndex = 0;
    read_bytes = fread(&vulkanDeviceIndex, 1, 4, stdin);
    if (read_bytes < 4)
    {
        fprintf(stderr, "read only %zd bytes, using default vulkanDeviceIndex\n", read_bytes);
        vulkanDeviceIndex = 0;
    }

    read_bytes = fread(workSize, 1, 12, stdin);
    if (read_bytes < 12)
    {
        fprintf(stderr, "read only %zd bytes, using default workSize\n", read_bytes);
         workSize[0] = workSize[1] = workSize[2] = 1;
    }

    inputBufferSize = 0;
    read_bytes = fread(&inputBufferSize, 1, 4, stdin);
    if (read_bytes < 4)
    {
        fprintf(stderr, "read only %zd bytes, using default inputBufferSize\n", read_bytes);
        inputBufferSize = 4;
    }

    input = (char *)malloc(sizeof(ispc::inputs) - 4 + inputBufferSize);
}

bool readInput()
{
	if (feof(stdin)) {
		return false;
	}
	
    ::size_t input_length = 0, read_bytes = 0;
    ::size_t off = sizeof(ispc::inputs) - 4;

    while (input_length < inputBufferSize && !feof(stdin))
    {
        read_bytes = fread((void *)(input + input_length + off), 1, inputBufferSize, stdin);
        input_length += read_bytes;
    }
    return input_length > 0;
}

EMSCRIPTEN_KEEPALIVE extern "C" 
int run(int w, int h, int d, ispc::inputs *inputs, ispc::outputs *outputs) {
  workSize[0] = w;
  workSize[1] = h;
  workSize[2] = d;
  ispc::runner_main(workSize, *inputs, *outputs);
  return (int)(outputs->outputData);
}

int main(int argc, char *argv[])
{
    return 0;

    EM_ASM({
      console.time('compute');
    });
    inputBufferSize = 8*4;
    input = (char *)malloc(sizeof(ispc::inputs) - 4 + inputBufferSize);
    float *dims = (float*)(input + (sizeof(ispc::inputs) - 4));
    dims[0] = 1920;
    dims[1] = 1080;
    dims[2] = 0;
    dims[3] = 0;
    dims[4] = 0;
    dims[5] = 0;
    dims[6] = 0;
    dims[7] = 0;

    bufferSize = dims[0]*dims[1]*4;
    
    ispc::outputs *outputs = (ispc::outputs *)malloc(sizeof(ispc::outputs) - 4 + bufferSize);
    ispc::inputs *inputs = (ispc::inputs *)input;

    int output_ptr = run(dims[0]/192, dims[1]/10, 1, inputs, outputs);

    EM_ASM({
	console.timeEnd('compute');
	var c = Module.canvas;
	var ctx = c.getContext('2d');
	c.width = $1; c.height = $2;
	var id = ctx.createImageData(c.width, c.height);
	var data = id.data;
	var off = $0;
	for (var i = 0; i < data.length; i++) {
	  data[i] = Module.HEAPU8[off + i];
	}
	ctx.putImageData(id, 0, 0);
    }, output_ptr, dims[0], dims[1]);

    free(input);
    free(outputs);
    
    return 0;
}

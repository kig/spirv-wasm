/*
 * Copyright 2015-2017 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#include <unistd.h>
#endif

#include "spirv_cross/external_interface.h"
#include <stdio.h>
#include <string.h>

#ifndef GLM_FORCE_SWIZZLE
#define GLM_FORCE_SWIZZLE
#endif

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/glm.hpp>
using namespace glm;

static const uint requestCount = 100000;

static uint inputBuffer[256 * requestCount] = {};
static uint outputBuffer[256 * requestCount] = {};
static uint heapBuffer[1024 * requestCount] = {};
static uint requestBuffer[1024 * requestCount] = {};
static uint responseBuffer[1024 * requestCount] = {};


int main()
{
#ifdef __EMSCRIPTEN__
	EM_ASM({
	    console.time('compute');
	});
#endif
	// First, we get the C interface to the shader.
	// This can be loaded from a dynamic library, or as here,
	// linked in as a static library.
	auto *iface = spirv_cross_get_interface();

	// Create an instance of the shader interface.
	auto *shader = iface->construct();

// Build some input data for our compute shader.
#define NUM_WORKGROUPS_X 50
#define NUM_WORKGROUPS_Y 100

	int bytes = fread((char*)inputBuffer, 1, 1024, stdin);
	for (int i = 1; i < requestCount; i++) {
		memcpy((void*)(inputBuffer + 256 * i), (void*)inputBuffer, bytes);
	}

	void *inputs_ptr = inputBuffer;
	void *outputs_ptr = outputBuffer;
	void *heap_ptr = heapBuffer;
	void *requests_ptr = requestBuffer;
	void *responses_ptr = responseBuffer;

	// Bind resources to the shader.
	// For resources like samplers and buffers, we provide a list of pointers,
	// since UBOs, SSBOs and samplers can be arrays, and can point to different types,
	// which is especially true for samplers.
	spirv_cross_set_resource(shader, 0, 0, &inputs_ptr, sizeof(inputs_ptr));
	spirv_cross_set_resource(shader, 0, 1, &outputs_ptr, sizeof(outputs_ptr));
	spirv_cross_set_resource(shader, 0, 2, &heap_ptr, sizeof(heap_ptr));
	spirv_cross_set_resource(shader, 0, 3, &requests_ptr, sizeof(requests_ptr));
	spirv_cross_set_resource(shader, 0, 4, &responses_ptr, sizeof(responses_ptr));

	// We also have to set builtins.
	// The relevant builtins will depend on the shader,
	// but for compute, there are few builtins, which are gl_NumWorkGroups and gl_WorkGroupID.
	// LocalInvocationID and GlobalInvocationID are inferred when executing the invocation.
	uvec3 num_workgroups(NUM_WORKGROUPS_X, NUM_WORKGROUPS_Y, 1);
	uvec3 work_group_id(0, 0, 0);
	spirv_cross_set_builtin(shader, SPIRV_CROSS_BUILTIN_NUM_WORK_GROUPS, &num_workgroups, sizeof(num_workgroups));
	spirv_cross_set_builtin(shader, SPIRV_CROSS_BUILTIN_WORK_GROUP_ID, &work_group_id, sizeof(work_group_id));

	// Execute work groups.
	for (unsigned x = 0; x < NUM_WORKGROUPS_X; x++)
	for (unsigned y = 0; y < NUM_WORKGROUPS_Y; y++)
	{
		work_group_id.x = x;
		work_group_id.y = y;
		iface->invoke(shader);
	}

	// Call destructor.
	iface->destruct(shader);

#ifdef __EMSCRIPTEN__
	EM_ASM({
	    console.timeEnd('compute');
	    var c = Module.canvas;
	    var ctx = c.getContext('2d');
	    c.width = c.height = 1280;
	    var id = ctx.createImageData(c.width, c.height);
	    var data = id.data;
            var off = $0 / 4;
	    for (var i = 0; i < data.length; i++) {
	      data[i] = (Module.HEAPF32[off + i] * 255.0) | 0;
	    }
	    ctx.putImageData(id, 0, 0);
	}, (int)outputs_ptr);
#else
	for (int i = 0; i < 1; i++) {
		write(1, ((char*)outputs_ptr)+4+1024*i, outputBuffer[256*i]);
	}
#endif

	return 0;
}

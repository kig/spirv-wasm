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

#include <unistd.h>

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

// Build some input data for our compute shader.
#define NUM_WORKGROUPS_X 32
#define NUM_WORKGROUPS_Y 1

static const uint requestCount = NUM_WORKGROUPS_X * NUM_WORKGROUPS_Y * 1024 * 16;

static const int requestSize = 1024;

static int inputBuffer[(requestSize / 4) * requestCount] = {};
static int outputBuffer[(requestSize / 4) * requestCount] = {};
static int heapBuffer[(requestSize / 4) * requestCount] = {};

int main()
{
	// First, we get the C interface to the shader.
	// This can be loaded from a dynamic library, or as here,
	// linked in as a static library.
	auto *iface = spirv_cross_get_interface();

	// Create an instance of the shader interface.
	auto *shader = iface->construct();

	void *inputs_ptr = inputBuffer;
	void *outputs_ptr = outputBuffer;
	void *heap_ptr = heapBuffer;

	int requestTemplate[(requestSize / 4)];
	for (int i = 0; i < requestCount; i++) {
		if (i % 2 == 0) {
			snprintf((char*)(&inputBuffer[(requestSize / 4) * i + 4]), ((requestSize / 16) - 1) * 16, "POST /%07d HTTP/1.1\r\nhost: localhost\r\n\r\ntext/html\r\n\r\n<html><body>This is post number %d.</body></html>", i*2/3, i);
		} else {
			snprintf((char*)(&inputBuffer[(requestSize / 4) * i + 4]), ((requestSize / 16) - 1) * 16, "GET /%07d HTTP/1.1\r\nhost: localhost\r\n\r\n", i);
		}
		inputBuffer[(requestSize / 4) * i] = strlen((char*)(&inputBuffer[(requestSize / 4) * i + 4]));
		// if (i < 10) printf("%d\n%s\n", inputBuffer[(requestSize / 4) * i], (char*)(&inputBuffer[(requestSize / 4) * i + 4]));

		snprintf((char*)(&heapBuffer[(requestSize / 4) * i + 4]), ((requestSize / 16) - 1) * 16, "text/html\r\n\r\n<html><body>This is document number %d.</body></html>", i);
		heapBuffer[(requestSize / 4) * i] = strlen((char*)(&heapBuffer[(requestSize / 4) * i + 4]));
		// if (i < 10) printf("%d\n%s\n", heapBuffer[(requestSize / 4) * i], (char*)(&heapBuffer[(requestSize / 4) * i + 4]));
	}

	for (int i = 0; i < 1000; i++) {

		// Bind resources to the shader.
		// For resources like samplers and buffers, we provide a list of pointers,
		// since UBOs, SSBOs and samplers can be arrays, and can point to different types,
		// which is especially true for samplers.
		spirv_cross_set_resource(shader, 0, 0, &inputs_ptr, sizeof(inputs_ptr));
		spirv_cross_set_resource(shader, 0, 1, &outputs_ptr, sizeof(outputs_ptr));
		spirv_cross_set_resource(shader, 0, 2, &heap_ptr, sizeof(heap_ptr));

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

	}

	// Call destructor.
	iface->destruct(shader);

	for (int i = 0; i < 10; i++) {
		write(1, ((char*)outputBuffer)+requestSize*i+16, outputBuffer[(requestSize / 4)*i]);
		printf("\n");
	}

	return 0;
}

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
#include "httpd.ispc.h"

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
using namespace ispc;

// Build some input data for our compute shader.
#define NUM_WORKGROUPS_X 32
#define NUM_WORKGROUPS_Y 32

static const uint requestCount = NUM_WORKGROUPS_X * NUM_WORKGROUPS_Y * 32;

static uint8_t inputs[1024 * requestCount] = {};
static uint8_t outputs[1024 * requestCount] = {};
static uint8_t heaps[1024 * requestCount] = {};

int main()
{
	int bytes = fread(((char*)inputs), 1, 1024-16, stdin);
	((uint32_t*)inputs)[0] = bytes;
	for (int i = 1; i < requestCount; i++) {
		memcpy((void*)(inputs + 1024 * i), (void*)inputs, 1024);
	}
	for (int j = 0; j < 1000; j++) {
		int32_t workgroups[] = {NUM_WORKGROUPS_X, NUM_WORKGROUPS_Y, 1};
		runner_main(workgroups,
			*(struct inputBuffer*)inputs,
			*(struct outputBuffer*)outputs,
			*(struct heapBuffer*)heaps
		);
	}
	for (int i = 0; i < 1; i++) {
		printf("%d\n", ((uint32_t*)outputs)[256*i]);
		write(1, outputs+1024*i, ((uint32_t*)outputs)[256*i]);
	}

	return 0;
}

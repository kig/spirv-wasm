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
#include <chrono>

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
#define NUM_WORKGROUPS_Y 1

static const uint requestCount = NUM_WORKGROUPS_X * NUM_WORKGROUPS_Y * 16 * 512;

static uint8_t inputs[1024 * requestCount] = {};
static uint8_t outputs[1024 * requestCount] = {};
static uint8_t heaps[1024 * requestCount] = {};

int main()
{
	int bytes = fread(((char*)inputs)+16, 1, 1024-16, stdin);
	((uint32_t*)inputs)[0] = bytes;
	for (int i = 1; i < requestCount; i++) {
		memcpy((void*)(inputs + 1024 * i), (void*)inputs, 1024);
	}

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	for (int j = 0; j < 1000; j++) {
		int32_t workgroups[] = {NUM_WORKGROUPS_X, NUM_WORKGROUPS_Y, 1};
		runner_main(workgroups,
			*(struct inputBuffer*)inputs,
			*(struct outputBuffer*)outputs,
			*(struct heapBuffer*)heaps
		);
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	for (int i = 0; i < 1; i++) {
		printf("%d\n", ((uint32_t*)outputs)[256*i]);
		write(1, outputs+1024*i+16, ((uint32_t*)outputs)[256*i]);
	}

	printf("Elapsed: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
	printf("Million requests per second: %.3f\n", 1e-6 * (requestCount * 1000.0) / (0.001 * std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()));

	return 0;
}

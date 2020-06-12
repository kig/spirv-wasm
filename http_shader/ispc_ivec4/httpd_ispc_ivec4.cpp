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

static const uint requestCount = NUM_WORKGROUPS_X * NUM_WORKGROUPS_Y * 1024 * 16;

static const int requestSize = 1024;

static int inputBuffe[(requestSize / 4) * requestCount] = {};
static int outputBuffe[(requestSize / 4) * requestCount] = {};
static int heapBuffe[(requestSize / 4) * requestCount] = {};

int main()
{
	int requestTemplate[(requestSize / 4)];
	for (int i = 0; i < requestCount; i++) {
		if (i % 2 == 0) {
			snprintf((char*)(&inputBuffe[(requestSize / 4) * i + 4]), ((requestSize / 16) - 1) * 16, "POST /%07d HTTP/1.1\r\nhost: localhost\r\n\r\ntext/html\r\n\r\n<html><body>This is post number %d.</body></html>", i*2/3, i);
		} else {
			snprintf((char*)(&inputBuffe[(requestSize / 4) * i + 4]), ((requestSize / 16) - 1) * 16, "GET /%07d HTTP/1.1\r\nhost: localhost\r\n\r\n", i);
		}
		if (i % 11 == 10) {
			int j = i % 10;
			snprintf((char*)(&inputBuffe[(requestSize / 4) * i + 4]), ((requestSize / 16) - 1) * 16, "POST /%07d HTTP/1.1\r\nhost: localhost\r\n\r\ntext/html\r\n\r\n<html><body>This is %d spam-post %d number %d.</body></html>", j, i, i, i);
		}
		inputBuffe[(requestSize / 4) * i] = strlen((char*)(&inputBuffe[(requestSize / 4) * i + 4]));
		// if (i < 10) printf("%d\n%s\n", inputBuffe[(requestSize / 4) * i], (char*)(&inputBuffe[(requestSize / 4) * i + 4]));

		snprintf((char*)(&heapBuffe[(requestSize / 4) * i + 4]), ((requestSize / 16) - 1) * 16, "text/html\r\n\r\n<html><body>This is document number %d.</body></html>", i);
		heapBuffe[(requestSize / 4) * i] = strlen((char*)(&heapBuffe[(requestSize / 4) * i + 4]));
		// if (i < 10) printf("%d\n%s\n", heapBuffe[(requestSize / 4) * i], (char*)(&heapBuffe[(requestSize / 4) * i + 4]));
	}

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	for (int j = 0; j < 1000; j++) {
		int32_t workgroups[] = {NUM_WORKGROUPS_X, NUM_WORKGROUPS_Y, 1};
		runner_main(workgroups,
			*(struct inputBuffer*)inputBuffe,
			*(struct outputBuffer*)outputBuffe,
			*(struct heapBuffer*)heapBuffe
		);
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	for (int i = 0; i < 10; i++) {
		write(1, ((char*)outputBuffe)+requestSize*i+16, outputBuffe[(requestSize / 4)*i]);
		printf("\n");
	}

	printf("\nElapsed: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
	printf("Million requests per second: %.3f\n\n", 1e-6 * (requestCount * 1000.0) / (0.001 * std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()));


	return 0;
}

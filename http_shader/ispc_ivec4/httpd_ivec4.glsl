#define version #version

version 450

#include "../chr.glsl"

#define REQUEST_SIZE 1024
#define RESPONSE_SIZE 1024
#define HEAP_SIZE 1024

#define REQUESTS_PER_INVOCATION 1024

#define HEAP_TOTAL_SZ (32 * 16 * 1024 * (HEAP_SIZE / 16))

layout ( local_size_x = 16, local_size_y = 1, local_size_z = 1 ) in;

layout(std430, binding = 0) readonly buffer inputBuffer { highp ivec4 inputBytes[]; };
layout(std430, binding = 1) buffer outputBuffer { highp ivec4 outputBytes[]; };
layout(std430, binding = 2) buffer heapBuffer { highp ivec4 heap[]; };

const highp int METHOD_GET = 'GET ';
const highp int METHOD_POST = 'POST';
const highp int METHOD_OPTION = 'OPTI';
const highp int METHOD_UNKNOWN = 0;

void setE(inout ivec4 v, int i, int value) {
	if (i == 0) v.x = value;
	else if (i == 1) v.y = value;
	else if (i == 2) v.z = value;
	else v.w = value;
}

int getE(ivec4 v, int i) {
	int value = v.x;
	if (i == 1) value = v.y;
	else if (i == 2) value = v.z;
	else if (i == 3) value = v.w;
	return value;
}

void main() {
	int wgId = int(gl_GlobalInvocationID.x) * REQUESTS_PER_INVOCATION;

	for (int j = 0; j < REQUESTS_PER_INVOCATION; j++) {
		int reqOff = (wgId+j) * (REQUEST_SIZE / 16);
		int resOff = (wgId+j) * (RESPONSE_SIZE / 16);

		// Parse request in format
		// [GET |/xxx|xxxx| HTT|P/1.|1\r\n.|...]
		// [POST| /xx|xxxx|x HT|TP/1|.1\r\n|....\r\nmimetype\r\n\r\npost body]
		ivec4 requestInfo = inputBytes[reqOff];
		if (requestInfo.x == 0) { // skip empty requests
			continue;
		}
		ivec4 req = inputBytes[reqOff+1];
		ivec4 req2 = inputBytes[reqOff+2];
		int method = req.x;

		int i = resOff;

		if (method == METHOD_GET) {
			// Parse key from path /xxxxxxx
			int key = (
				(((req.y >> 8) & 0xFF) - 48) * 1000000 +
				(((req.y >> 16) & 0xFF) - 48) * 100000 +
				(((req.y >> 24) & 0xFF) - 48) * 10000 +
				(((req.z >> 0) & 0xFF) - 48) * 1000 +
				(((req.z >> 8) & 0xFF) - 48) * 100 +
				(((req.z >> 16) & 0xFF) - 48) * 10 +
				(((req.z >> 24) & 0xFF) - 48) * 1
			) * (HEAP_SIZE / 16);
			// Check that the key is valid and fetch the content from the heap buffer if so.
			if (key >= 0 && key < HEAP_TOTAL_SZ && heap[key].x > 0 && heap[key].x <= RESPONSE_SIZE - 3 * 16) {
				outputBytes[i+0] = ivec4(2*16 + heap[key].x, 0, 0, 0);
				outputBytes[i+1] = ivec4('200 ', 'OK H', 'TTP/', '1.1\r');
				outputBytes[i+2] = ivec4('\ncon', 'tent', '-typ', 'e:  ');
				int len = heap[key].x / 16 + (heap[key].x % 16 > 0 ? 1 : 0);
				for (int k = 0; k < len; k++) {
					outputBytes[i+3+k] = heap[key+1+k];
				}
				continue;
			}
		} else if (method == METHOD_POST) {
			// Parse key from path /xxxxxxx
			int key = (
				(((req.y >> 16) & 0xFF) - 48) * 1000000 +
				(((req.y >> 24) & 0xFF) - 48) * 100000 +
				(((req.z >> 0) & 0xFF) - 48) * 10000 +
				(((req.z >> 8) & 0xFF) - 48) * 1000 +
				(((req.z >> 16) & 0xFF) - 48) * 100 +
				(((req.z >> 24) & 0xFF) - 48) * 10 +
				(((req.w >> 0) & 0xFF) - 48) * 1
			) * (HEAP_SIZE / 16);
			// If the key is valid, replace the content in the heap buffer with the post body.
			if (key >= 0 && key < HEAP_TOTAL_SZ) {
				int rnrn = 0;
				int readStart = 0;
				int readEnd = 512;
				ivec4 w = ivec4(0);
				int l = 0;
				int hi = 0;
				for (int k = 13; k < REQUEST_SIZE && k < HEAP_SIZE; k++) {
					int v4i = k / 16;
					int vi = k - (v4i * 16);
					int c = vi / 4;
					int b = vi - (c * 4);
					int chr = (getE(inputBytes[reqOff + 1 + v4i], c) >> (b * 8)) & 0xFF;
					if (readStart > 0) {
						if (chr == 0) {
							readEnd = k;
							break;
						}
						int wc = l / 4;
						int wb = l - (wc * 4);
						setE(w, wc, getE(w, wc) | (chr << (wb * 8)));
						l++;
						if (l == 16) {
							heap[key+1+hi] = w;
							hi++;
							w *= 0;
							l = 0;
						}
					} else if (chr == CHR_CR && (rnrn & 1) == 0) {
						rnrn++;
					} else if (chr == CHR_LF && (rnrn & 1) == 1) {
						rnrn++;
						if (rnrn == 4) {
							readStart = k;
						}
					} else {
						rnrn = 0;
					}
				}
				if (l > 0 && (1 + hi) < (HEAP_SIZE/16)) {
					heap[key+1+hi] = w;
				}
				heap[key].x = readEnd - readStart;
				outputBytes[i+0] = ivec4(3*16, 0, 0, 0);
				outputBytes[i+1] = ivec4('200 ', 'OK H', 'TTP/', '1.1\r');
				outputBytes[i+2] = ivec4('\ncon', 'tent', '-typ', 'e: t');
				outputBytes[i+3] = ivec4('ext/', 'plai', 'n\r\n\r', '\nOK.');
				continue;
			}
		}
		outputBytes[i+0] = ivec4(3*16 - 3*4 - 2, 0, 0, 0);
		outputBytes[i+1] = ivec4('500 ', 'BAD ', 'HTTP', '/1.1');
		outputBytes[i+2] = ivec4('\r\n\r\n', req.x, req.y, req.z);
		outputBytes[i+3] = ivec4(req.w, req2.x, req2.y, req2.z);
		
	}

}

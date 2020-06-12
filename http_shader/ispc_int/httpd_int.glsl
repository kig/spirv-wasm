#define version #version

version 450

#include "../chr.glsl"

#define STRIDE 32

#define BSZ 1024

#define heapStrCopy(str, SRC, DST, i, index) {int _s = (str).x; int _e = (str).y; while (_s < _e) {(DST)[index+i*STRIDE] = (SRC)[index+_s*STRIDE]; i++; _s++;}}
#define strCopySlice(SRC, DST, i, index, start, end) {int _s = start; int _e = end; while (_s < _e) {(DST)[index+i*STRIDE] = (SRC)[_s++]; i++;}}
#define strCopy(SRC, DST, i, index) {int _str[] = SRC; strCopySlice(_str, DST, i, index, 0, _str.length())}
#define W(chr) response[index + i*STRIDE] = (chr); i++;

#define A_OK if (i > BSZ) { return error(index); }

layout ( local_size_x = STRIDE, local_size_y = 1, local_size_z = 1 ) in;

layout(std430, binding = 0) readonly buffer inputBuffer { highp int inputBytes[]; };
layout(std430, binding = 1) buffer outputBuffer { highp int outputBytes[]; };
layout(std430, binding = 2) buffer heapBuffer { lowp int heap[]; };
layout(std430, binding = 3) buffer requestBuffer { lowp int request[]; };
layout(std430, binding = 4) buffer responseBuffer { lowp int response[]; };

const highp int METHOD_GET = 'GET ';
const highp int METHOD_POST = 'POST';
const highp int METHOD_OPTION = 'OPTI';
const highp int METHOD_UNKNOWN = 0;

const lowp int PROTOCOL_UNKNOWN = 0;
const lowp int PROTOCOL_HTTP10 = '/1.0';
const lowp int PROTOCOL_HTTP11 = '/1.1';

const lowp int MIME_TEXT_PLAIN = 0;
const lowp int MIME_TEXT_HTML = 1;


int strLen(ivec2 str) {
	return str.y - str.x;
}


struct header {
	ivec2 name;
	ivec2 value;
};

void readRequestUntilChar(inout int i, int index, int endChar, out ivec2 str) {
	str.x = i;
	while (i < BSZ && request[index+i*STRIDE] != endChar) {
		i++;
	}
	str.y = i;
	i++;
}

void readMethod(inout int i, int index, out int method) {
	int j = index + i * STRIDE;
	if (
		request[j+0*STRIDE] == CHR_G &&
		request[j+1*STRIDE] == CHR_E &&
		request[j+2*STRIDE] == CHR_T &&
		request[j+3*STRIDE] == CHR_SPACE
	) {
		method = METHOD_GET;
		i += 4;
		return;
	} else if (
		request[j+0*STRIDE] == CHR_P &&
		request[j+1*STRIDE] == CHR_O &&
		request[j+2*STRIDE] == CHR_S &&
		request[j+3*STRIDE] == CHR_T &&
		request[j+4*STRIDE] == CHR_SPACE
	) {
		method = METHOD_POST;
		i += 5;
		return;
	} else if (request[j+0*STRIDE] == CHR_O && request[j+6*STRIDE] == CHR_SPACE) {
		method = METHOD_OPTION;
		i += 7;
		return;
	}
	method = METHOD_UNKNOWN;
	i = BSZ+1;
}

void readPath(inout int i, int index, out ivec2 path) {
	readRequestUntilChar(i, index, CHR_SPACE, path);
}

void readProtocol(inout int i, int index, out int protocol) {
	ivec2 protocolString;
	readRequestUntilChar(i, index, CHR_CR, protocolString);
	if (i < BSZ && request[index+i*STRIDE] == CHR_LF) {
		i++;
		if (request[index+(protocolString.y-1)*STRIDE] == CHR_1) {
			protocol = PROTOCOL_HTTP11;
		} else {
			protocol = PROTOCOL_HTTP10;
		}
	} else {
		protocol = PROTOCOL_UNKNOWN;
		i = BSZ+1;
	}
}

bool readHeader(inout int i, int index, out header hdr) {
	if (request[index+i*STRIDE] == CHR_CR) {
		i += 2;
		return true;
	}
	readRequestUntilChar(i, index, CHR_COLON, hdr.name);
	while (i < BSZ && request[index+i*STRIDE] == CHR_SPACE) i++;
	readRequestUntilChar(i, index, CHR_CR, hdr.value);
	i++;
	return false;
}

void writeCRLF(inout int i, int index) {
	W(CHR_CR);
	W(CHR_LF);
}

void writeStatus(inout int i, int index, int statusCode) {
	strCopy("HTTP/1.1 ", response, i, index);
	if (statusCode == 200) {
		strCopy("200 OK", response, i, index);
	} else {
		strCopy("500 Error", response, i, index);
	}
	writeCRLF(i, index);
}

void writeContentType(inout int i, int index, int contentType) {
	int contentTypeString[] = "Content-Type: ";
	strCopy(contentTypeString, response, i, index);
	if (contentType == MIME_TEXT_PLAIN) {
		strCopy("text/plain", response, i, index);
	} else {
		strCopy("text/html", response, i, index);
	}
	writeCRLF(i, index);
}

void writeBody(inout int i, int index, ivec2 path, header headers[32], int headerCount) {
	strCopy("Hello, World!", response, i, index);
	W(CHR_LF);
	for (int j = 0; j < 32; j++) {
		if (j >= headerCount) break;
		ivec2 name = headers[j].name;
		ivec2 value = headers[j].value;
		if (strLen(name) + 3 + strLen(value) + i > 1023) break;
		heapStrCopy(name, request, response, i, index);
		strCopy(": ", response, i, index);
		heapStrCopy(value, request, response, i, index);
		W(CHR_LF);
	}
}

int error(int index) {
	int i = 0;
	writeStatus(i, index, 500);
	writeContentType(i, index, MIME_TEXT_PLAIN);
	writeCRLF(i, index);
	return i;
}

void unpackRequest(int byteIndex, int index) {
	int len = inputBytes[byteIndex];
	for (int j = 0; j < min(256, len/4+1); j++) {
		int v = inputBytes[byteIndex + j + 1];
		int off = index + (j * 4) * STRIDE;
		request[off + 0*STRIDE] = (v >> 0) & 0xFF;
		request[off + 1*STRIDE] = (v >> 8) & 0xFF;
		request[off + 2*STRIDE] = (v >> 16) & 0xFF;
		request[off + 3*STRIDE] = (v >> 24) & 0xFF;
	}
}

void packResponse(int byteIndex, int index, int len) {
	outputBytes[byteIndex] = len;
	for (int j = 1; j < min(256, len/4+1); j++) {
		int off = index + (j * 4 - 4) * STRIDE;
		ivec4 v = ivec4(
			((response[off + 0*STRIDE] & 0xFF) << 0),
		    ((response[off + 1*STRIDE] & 0xFF) << 8),
			((response[off + 2*STRIDE] & 0xFF) << 16),
			((response[off + 3*STRIDE] & 0xFF) << 24)
		);
		outputBytes[byteIndex + j] = (v.x | v.y | v.z | v.w);
	}
}

int handleRequest(int index) {
	int method;
	ivec2 path;
	int protocol;
	header headers[32];
	int headerCount = 0;

	int i = 0;
	readMethod(i, index, method);
	readPath(i, index, path);
	readProtocol(i, index, protocol);
	for (int j = 0; j < 32; j++) {
		if (readHeader(i, index, headers[j])) {
			break;
		}
		headerCount++;
	}
	A_OK;

	i = 0;
	writeStatus(i, index, 200);
	writeContentType(i, index, MIME_TEXT_PLAIN);
	writeCRLF(i, index);
	writeBody(i, index, path, headers, headerCount);
	return i;
}

void main() {
	int wgId = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x));
	int index = STRIDE * BSZ * (wgId / STRIDE);
	index += wgId & (STRIDE-1);
	unpackRequest(wgId*(BSZ/4), index);
	int len = handleRequest(index);
	packResponse(wgId*(BSZ/4), index, len);
}

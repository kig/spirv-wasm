#define version #version

version 450

#include "../chr.glsl"

#define strCopy(SRC, DST, i, start, end) uint _s = start; uint _e = end; while (_s < _e) (DST)[i++] = (SRC)[_s++];
#define strCopyAll(SRC, DST, i) uint _str[] = SRC; strCopy(_str, DST, i, 0, _str.length())

#define A_OK if (i > 1024) { error(index); return; }

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

layout(std430, binding = 0) readonly buffer inputBuffer { uint inputBytes[]; };
layout(std430, binding = 1) buffer outputBuffer { uint outputBytes[]; };
layout(std430, binding = 2) buffer heapBuffer { uint heap[]; };
layout(std430, binding = 3) buffer requestBuffer { uint request[]; };
layout(std430, binding = 4) buffer responseBuffer { uint response[]; };


const uint METHOD_UNKNOWN = 0;
const uint METHOD_GET = 1;
const uint METHOD_POST = 2;
const uint METHOD_OPTION = 3;

const uint PROTOCOL_UNKNOWN = 0;
const uint PROTOCOL_HTTP10 = 1;
const uint PROTOCOL_HTTP11 = 2;

const uint MIME_TEXT_PLAIN = 0;
const uint MIME_TEXT_HTML = 1;

struct header {
	uvec2 name;
	uvec2 value;
};

void readRequestUntilChar(inout uint i, uint index, uint endChar, out uvec2 str) {
	str.x = index + i;
	while (i < 1024 && request[index+i] != endChar) {
		i++;
	}
	str.y = index + i;
	i++;
}

void readMethod(inout uint i, uint index, out uint method) {
	uint j = index + i;
	uint c = request[j];
	if (
		request[j] == CHR_G &&
		request[j+1] == CHR_E &&
		request[j+2] == CHR_T &&
		request[j+3] == CHR_SPACE
	) {
		method = METHOD_GET;
		i += 4;
		return;
	} else if (
		request[j] == CHR_P &&
		request[j+1] == CHR_O &&
		request[j+2] == CHR_S &&
		request[j+3] == CHR_T &&
		request[j+4] == CHR_SPACE
	) {
		method = METHOD_POST;
		i += 5;
		return;
	} else if (request[j] == CHR_O && request[j+6] == CHR_SPACE) {
		method = METHOD_OPTION;
		i += 7;
		return;
	}
	method = METHOD_UNKNOWN;
	i = 1025;
}

void readPath(inout uint i, uint index, out uvec2 path) {
	readRequestUntilChar(i, index, CHR_SPACE, path);
}

void readProtocol(inout uint i, uint index, out uint protocol) {
	uvec2 protocolString;
	readRequestUntilChar(i, index, CHR_CR, protocolString);
	if (i < 1024 && request[index+i] == CHR_LF) {
		i++;
		if (request[protocolString.y-1] == CHR_1) {
			protocol = PROTOCOL_HTTP11;
		} else {
			protocol = PROTOCOL_HTTP10;
		}
	} else {
		protocol = PROTOCOL_UNKNOWN;
		i = 1025;
	}
}

bool readHeader(inout uint i, uint index, out header hdr) {
	if (request[index+i] == CHR_CR) {
		i += 2;
		return true;
	}
	readRequestUntilChar(i, index, CHR_COLON, hdr.name);
	while (i < 1024 && request[index+i] == CHR_SPACE) i++;
	readRequestUntilChar(i, index, CHR_CR, hdr.value);
	i++;
	return false;
}

void writeStatus(inout uint i, uint index, uint statusCode) {
	uint j = i + index;
	strCopyAll("HTTP/1.1 ", response, j);
	if (statusCode == 200) {
		strCopyAll("200 OK", response, j);
	} else {
		strCopyAll("500 Error", response, j);
	}
	response[j++] = CHR_CR;
	response[j++] = CHR_LF;
	i = j - index;
}

void writeContentType(inout uint i, uint index, uint contentType) {
	uint j = i + index;

	uint contentTypeString[] = "Content-Type: ";
	strCopyAll(contentTypeString, response, j);
	if (contentType == MIME_TEXT_PLAIN) {
		strCopyAll("text/plain", response, j);
	} else {
		strCopyAll("text/html", response, j);
	}
	response[j++] = CHR_CR;
	response[j++] = CHR_LF;

	i = j - index;
}

void writeEndHeaders(inout uint i, uint index) {
	uint j = i + index;
	response[j++] = CHR_CR;
	response[j++] = CHR_LF;
	i = j - index;
}

void writeBody(inout uint i, uint index, uvec2 path) {
	uint j = i + index;
	strCopyAll("Hello, World!", response, j);
	response[j++] = CHR_LF;
	i = j - index;
}

void error(uint index) {
	uint i = 0;
	writeStatus(i, index, 500);
	writeContentType(i, index, MIME_TEXT_PLAIN);
	writeEndHeaders(i, index);
	response[index + 1023] = i;
}

void unpackRequest(uint index) {
	uint len = inputBytes[index/4 + 255];
	for (uint j = 0; j < len/4+1; j++) {
		uint v = inputBytes[index/4 + j];
		uint off = index + j * 4;
		request[off + 0] = v & 0xFF;
		request[off + 1] = (v >> 8) & 0xFF;
		request[off + 2] = (v >> 16) & 0xFF;
		request[off + 3] = v >> 24;
	}
}

void packResponse(uint index) {
	outputBytes[index/4 + 255] = response[index + 1023];
	uint len = response[index + 1023];
	for (uint j = 0; j < len+1; j++) {
		uint off = index + j * 4;
		outputBytes[index/4 + j] = (
			(response[off + 0] & 0xFF) |
			((response[off + 1] & 0xFF) << 8) |
			((response[off + 2] & 0xFF) << 16) |
			((response[off + 3] & 0xFF) << 24)
		);
	}
}

void handleRequest(uint index) {
	uint i = 0;
	uint method;
	uvec2 path;
	uint protocol;
	header headers[32];
	uint headerCount = 0;

	readMethod(i, index, method); A_OK;
	readPath(i, index, path); A_OK;
	readProtocol(i, index, protocol); A_OK;

	for (uint j = 0; j < 32; j++) {
		bool done = readHeader(i, index, headers[j]); A_OK;
		if (done) break;
		headerCount++;
	}

	i = 0;
	writeStatus(i, index, 200);
	writeContentType(i, index, MIME_TEXT_PLAIN);
	writeEndHeaders(i, index);
	writeBody(i, index, path);
	response[index + 1023] = i;
}

void main() {
	uint index = 1024 * (gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x));
	unpackRequest(index);
	handleRequest(index);
	packResponse(index);
}

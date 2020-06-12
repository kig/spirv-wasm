#define version #version

version 450

#include "../chr.glsl"

#define BLK_SZ 1024

#define strCopy(SRC, DST, i, start, end) uint _s = start; uint _e = end; while (_s < _e) (DST)[i++] = (SRC)[_s++];
#define strCopyAll(SRC, DST, i) uint _str[] = SRC; strCopy(_str, DST, i, 0, _str.length())

#define A_OK if (i > BLK_SZ) { error(index); return; }

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

layout(std430, binding = 0) readonly buffer inputBuffer { lowp uint inputBytes[]; };
layout(std430, binding = 1) buffer outputBuffer { lowp uint outputBytes[]; };
layout(std430, binding = 2) buffer heapBuffer { lowp uint heap[]; };

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
	while (i < BLK_SZ && inputBytes[index+i] != endChar) {
		i++;
	}
	str.y = index + i;
	i++;
}

void readMethod(inout uint i, uint index, out uint method) {
	uint j = index + i;
	uint c = inputBytes[j];
	if (
		inputBytes[j] == CHR_G &&
		inputBytes[j+1] == CHR_E &&
		inputBytes[j+2] == CHR_T &&
		inputBytes[j+3] == CHR_SPACE
	) {
		method = METHOD_GET;
		i += 4;
		return;
	} else if (
		inputBytes[j] == CHR_P &&
		inputBytes[j+1] == CHR_O &&
		inputBytes[j+2] == CHR_S &&
		inputBytes[j+3] == CHR_T &&
		inputBytes[j+4] == CHR_SPACE
	) {
		method = METHOD_POST;
		i += 5;
		return;
	} else if (inputBytes[j] == CHR_O && inputBytes[j+6] == CHR_SPACE) {
		method = METHOD_OPTION;
		i += 7;
		return;
	}
	method = METHOD_UNKNOWN;
	i = BLK_SZ+1;
}

void readPath(inout uint i, uint index, out uvec2 path) {
	readRequestUntilChar(i, index, CHR_SPACE, path);
}

void readProtocol(inout uint i, uint index, out uint protocol) {
	uvec2 protocolString;
	readRequestUntilChar(i, index, CHR_CR, protocolString);
	if (i < 1024 && inputBytes[index+i] == CHR_LF) {
		i++;
		if (inputBytes[protocolString.y-1] == CHR_1) {
			protocol = PROTOCOL_HTTP11;
		} else {
			protocol = PROTOCOL_HTTP10;
		}
	} else {
		protocol = PROTOCOL_UNKNOWN;
		i = BLK_SZ+1;
	}
}

bool readHeader(inout uint i, uint index, out header hdr) {
	if (inputBytes[index+i] == CHR_CR) {
		i += 2;
		return true;
	}
	readRequestUntilChar(i, index, CHR_COLON, hdr.name);
	while (i < 1024 && inputBytes[index+i] == CHR_SPACE) i++;
	readRequestUntilChar(i, index, CHR_CR, hdr.value);
	i++;
	return false;
}

void writeStatus(inout uint i, uint index, uint statusCode) {
	uint j = i + index;
	strCopyAll("HTTP/1.1 ", outputBytes, j);
	if (statusCode == 200) {
		strCopyAll("200 OK", outputBytes, j);
	} else {
		strCopyAll("500 Error", outputBytes, j);
	}
	outputBytes[j++] = CHR_CR;
	outputBytes[j++] = CHR_LF;
	i = j - index;
}

void writeContentType(inout uint i, uint index, uint contentType) {
	uint j = i + index;

	uint contentTypeString[] = "Content-Type: ";
	strCopyAll(contentTypeString, outputBytes, j);
	if (contentType == MIME_TEXT_PLAIN) {
		strCopyAll("text/plain", outputBytes, j);
	} else {
		strCopyAll("text/html", outputBytes, j);
	}
	outputBytes[j++] = CHR_CR;
	outputBytes[j++] = CHR_LF;

	i = j - index;
}

void writeEndHeaders(inout uint i, uint index) {
	uint j = i + index;
	outputBytes[j++] = CHR_CR;
	outputBytes[j++] = CHR_LF;
	i = j - index;
}

void writeBody(inout uint i, uint index, uvec2 path) {
	uint j = i + index;
	strCopyAll("Hello, World!", outputBytes, j);
	outputBytes[j++] = CHR_LF;
	i = j - index;
}

void error(uint index) {
	uint i = 16;
	writeStatus(i, index, 500);
	writeContentType(i, index, MIME_TEXT_PLAIN);
	writeEndHeaders(i, index);
	outputBytes[index+0] = ((i-16) << 0) & 0xFF;
	outputBytes[index+1] = ((i-16) << 8) & 0xFF;
	outputBytes[index+2] = ((i-16) << 16) & 0xFF;
	outputBytes[index+3] = ((i-16) << 24) & 0xFF;
}

void handleRequest(uint index) {
	uint i = 16;
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

	i = 16;
	writeStatus(i, index, 200);
	writeContentType(i, index, MIME_TEXT_PLAIN);
	writeEndHeaders(i, index);
	writeBody(i, index, path);
	outputBytes[index+0] = ((i-16) << 0) & 0xFF;
	outputBytes[index+1] = ((i-16) << 8) & 0xFF;
	outputBytes[index+2] = ((i-16) << 16) & 0xFF;
	outputBytes[index+3] = ((i-16) << 24) & 0xFF;
}

void main() {
	uint index = BLK_SZ * (gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x));
	handleRequest(index);
}

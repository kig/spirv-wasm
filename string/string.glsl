#include "chr.glsl"

typedef string ivec2;
typedef stringArray ivec2;

int heapPtr = gl_GlobalInvocationID.x * HEAP_SIZE;

int strLen(ivec2 str) {
	return str.y - str.x;
}

string malloc(int len) {
	int ptr = heapPtr;
	heapPtr += len;
	return string(ptr, heapPtr);
}

int getC(string s, int index) {
	return heap[s.x + index];
}

void setC(string s, int index, int value) {
	heap[s.x + index] = value;
}

void strCopy(string dst, string src) {
	for (int i = dst.x, j = src.x; i < dst.y && j < src.y; i++, j++) {
		heap[i] = heap[j];
	}
}

string clone(string s) {
	string c = malloc(strLen(s));
	strCopy(c, s);
	return c;
}

bool isWhitespace(int c) {
	return (c == CHR_SPACE || c == CHR_TAB || c == CHR_LF || c == CHR_CR);
}

int lowercase(int c) {
	return c + (c >= CHR_A && c <= CHR_Z) ? (CHR_a - CHR_A) : 0;
}

int uppercase(int c) {
	return c + (c >= CHR_a && c <= CHR_z) ? (CHR_A - CHR_a) : 0;
}

void lowercase(string s) {
	for (int i = s.x; i < s.y; i++) {
		heap[i] = lowercase(heap[i]);
	}
}

void uppercase(string s) {
	for (int i = s.x; i < s.y; i++) {
		heap[i] = uppercase(heap[i]);
	}
}

void capitalize(string s) {
	bool afterSpace = true;
	for (int i = s.x; i < s.y; i++) {
		int c = heap[i];
		c += (afterSpace && c >= CHR_a && c <= CHR_z) ? (CHR_A - CHR_a) : 0;
		heap[i] = c;
		afterSpace = (c == ' ' || c == '\t' || c == '\r' || c == '\n');
	}	
}

void reverse(string s) {
	for (int i = s.x, j = s.y-1; i < j; i++, j--) {
		int tmp = heap[i];
		heap[i] = heap[j];
		heap[j] = tmp;
	}
}

int strCmp(string a, string b) {
	for (int i = a.x, j = b.x; i < a.y && j < b.y; i++, j++) {
		int d = heap[i] - heap[j];
		if (d != 0) return d;
	}
	return strLen(a) - strLen(b);
}

int strCmpI(string a, string b) {
	for (int i = a.x, j = b.x; i < a.y && j < b.y; i++, j++) {
		int d = lowercase(heap[i]) - lowercase(heap[j]);
		if (d != 0) return d;
	}
	return strLen(a) - strLen(b);
}

string concat(string a, string b) {
	string c = malloc(strLen(a) + strLen(b));
	int i = c.x;
	strCopy(c, a);
	strCopy(c + ivec2(strLen(a), 0), b);
	return c;
}

int indexOf(string s, int c) {
	for (int i = s.x; i < s.y; i++) {
		if (heap[i] == c) return i - s.x;
	}
	return -1;
}

int indexOf(string s, string key) {
	for (int i = s.x; i < s.y; i++) {
		if (heap[i] == heap[key.x]) {
			if (strCmp(string(i, s.y), key) == 0) {
				return i;
			}
		}
	}
	return -1;
}

int indexOfI(string s, string key) {
	for (int i = s.x; i < s.y; i++) {
		if (heap[i] == heap[key.x]) {
			if (strCmpI(string(i, s.y), key) == 0) {
				return i;
			}
		}
	}
	return -1;
}

int lastIndexOf(string s, int c) {
	for (int i = s.y-1; i >= s.x; i--) {
		if (heap[i] == c) return i - s.x;
	}
	return -1;
}

string slice(string s, int start, int end) {
	int len = strLen(s);
	start = normalizeIndex(start, len);
	end = normalizeIndex(end, len);
	return string(s.x + start, s.x + end);
}

ivec4 splitOnce(string s, int separator) {
	int idx = indexOf(s, separator);
	ivec4 pair;
	if (idx < 0) idx = s.y - s.x;
	pair.xy = string(s.x, s.x + idx);
	pair.zw = string(s.x + idx + 1, s.y);
	return pair;
}

ivec4 splitOnce(string s, string separator) {
	int idx = indexOf(s, separator);
	int len = strLen(separator);
	ivec4 pair;
	if (idx < 0) idx = s.y - s.x;
	pair.xy = string(s.x, s.x + idx);
	pair.zw = string(s.x + idx + len, s.y);
	return pair;
}

string readLine(inout int i, string s) {
	ivec4 pair = splitOnce(s + ivec2(i, 0), '\n');
	i = pair.z - s.x;
	return pair.xy;
}

stringArray split(string s, int separator) {
	int count = 0;
	int start = heapPtr;
	string rest = s;
	while (strLen(rest) >= 0) {
		ivec4 pair = splitOnce(rest, separator);
		rest = pair.zw;
		heap[heapPtr++] = pair.x;
		heap[heapPtr++] = pair.y;
		count++;
	}
	return stringArray(start, start+count);
}

stringArray split(string s, string separator) {
	int count = 0;
	int start = heapPtr;
	string rest = s;
	while (strLen(rest) >= 0) {
		ivec4 pair = splitOnce(rest, separator);
		rest = pair.zw;
		heap[heapPtr++] = pair.x;
		heap[heapPtr++] = pair.y;
		count++;
	}
	return stringArray(start, start+count);
}

string join(stringArray a, string joiner) {
	int start = heapPtr;
	int joinerLen = strLen(joiner);
	for (int i = a.x; i < a.y-2; i += 2) {
		string str = string(heap[i], heap[i+1]);
		int len = strLen(str);
		strCopy(string(heapPtr, heapPtr+len), str);
		heapPtr += len;
		len = joinerLen;
		strCopy(string(heapPtr, heapPtr+len), str);
		heapPtr += len;
	}
	if (a.y >= a.x) {
		string str = string(heap[a.y-2], heap[a.y-1]);
		int len = strLen(str);
		strCopy(string(heapPtr, heapPtr+len), str);
		heapPtr += len;
	}
	return string(start, heapPtr);
}

string join(ivec4 pair, string joiner) {
	int zwLen = strLen(pair.zw);
	if (zwLen < 0) {
		return pair.xy;
	}
	int xyLen = strLen(pair.xy);
	int joinerLen = strLen(joiner);
	string s = malloc(xyLen + joinerLen + zwLen);
	strCpy(s, pair.xy);
	strCpy(string(s.x + xyLen, s.y), joiner);
	strCpy(string(s.x + xyLen + joinerLen, s.y), pair.zw);
	return s;
}

string replaceOnce(string s, string pattern, string replacement) {
	int ptr = heapPtr;
	ivec4 pair = splitOnce(s, pattern);
	return join(pair, replacement);
}

string replace(string s, string pattern, string replacement) {
	int ptr = heapPtr;
	stringArray a = split(s, pattern);
	string res = join(a, replacement);
	
	string moved = string(ptr, ptr + strLen(res));
	strCpy(moved, res);
	heapPtr = moved.y;
	return moved;
}

string repeat(int char, int count) {
	string s = malloc(count);
	for (int i = s.x; i < s.y; i++) {
		heap[i] = char;
	}
	return s;
}

string repeat(string pattern, int count) {
	int len = strLen(pattern);
	string s = malloc(count * len);
	for (int i = s.x; i < s.y; i += len) {
		strCopy(string(i, s.y), pattern);
	}
	return s;
} 

string padStart(string s, int len, int filler) {
	int slen = strLen(s);
	int diff = len - slen;
	string res = malloc(max(slen, len));
	for (int i = 0; i < diff; i++) {
		setC(res, i, filler);
	}
	strCopy(res + ivec2(max(0,diff), 0), s);
	return res;
}

string padEnd(string s, int len, int filler) {
	int slen = strLen(s);
	int diff = len - slen;
	string res = malloc(max(slen, len));
	for (int i = 0; i < diff; i++) {
		setC(res, slen+i, filler);
	}
	strCopy(res, s);
	return res;
}

string truncate(string s, int maxLen) {
	return string(s.x, s.x + min(maxLen, strLen(s)));
}

string truncateEnd(string s, int maxLen) {
	return string(s.x + max(0, strLen(s) - maxLen), s.y);
}

bool startsWith(string s, string prefix) {
	return 0 == strCmp(truncate(s, strLen(prefix)), prefix);
}

bool endsWith(string s, string suffix) {
	return 0 == strCmp(truncateEnd(s, strLen(prefix)), suffix);
}

bool includes(string s, string key) {
	return indexOf(s, key) >= 0;
}

string trimStart(string s) {
	for (int i = s.x; i < s.y; i++) {
		int c = heap[i];
		if (!isWhitespace(c)) {
			s.x = i;
			break;
		}
	}
	return s;
}

string trimEnd(string s) {
	for (int i = s.y-1; i >= s.x; i--) {
		int c = heap[i];
		if (!isWhitespace(c)) {
			s.y = i;
			break;
		}
	}
	return s;
}

string trim(string s) {
	return trimEnd(trimStart(s));
}

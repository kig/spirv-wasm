#include "chr.glsl"

#define stringArray ivec2
#define string ivec2
#define char int8_t

int heapStart = int(gl_GlobalInvocationID.x) * HEAP_SIZE;
int heapEnd = heapStart + HEAP_SIZE;

int heapPtr = heapStart;

int i32heapPtr = heapStart;

int strLen(ivec2 str) {
	return str.y - str.x;
}

string malloc(int len) {
	int ptr = heapPtr;
	heapPtr += len;
	return string(ptr, heapPtr);
}

char getC(string s, int index) {
	return heap[s.x + index];
}

void setC(string s, int index, char value) {
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

bool isWhitespace(char c) {
	return (c == CHR_SPACE || c == CHR_TAB || c == CHR_LF || c == CHR_CR);
}

char lowercase(char c) {
	return c + char((c >= CHR_A && c <= CHR_Z) ? (CHR_a - CHR_A) : 0);
}

char uppercase(char c) {
	return c + char((c >= CHR_a && c <= CHR_z) ? (CHR_A - CHR_a) : 0);
}

string lowercase(string s) {
	for (int i = s.x; i < s.y; i++) {
		heap[i] = lowercase(heap[i]);
	}
	return s;	
}

string uppercase(string s) {
	for (int i = s.x; i < s.y; i++) {
		heap[i] = uppercase(heap[i]);
	}
	return s;	
}

string capitalize(string s) {
	bool afterSpace = true;
	for (int i = s.x; i < s.y; i++) {
		char c = heap[i];
		c += char((afterSpace && c >= CHR_a && c <= CHR_z) ? (CHR_A - CHR_a) : 0);
		heap[i] = c;
		afterSpace = (c == ' ' || c == '\t' || c == '\r' || c == '\n');
	}
	return s;	
}

string reverse(string s) {
	for (int i = s.x, j = s.y-1; i < j; i++, j--) {
		char tmp = heap[i];
		heap[i] = heap[j];
		heap[j] = tmp;
	}
	return s;	
}

int strCmp(string a, string b) {
	for (int i = a.x, j = b.x; i < a.y && j < b.y; i++, j++) {
		char d = heap[i] - heap[j];
		if (d != 0) return d;
	}
	return strLen(a) - strLen(b);
}

int strCmpI(string a, string b) {
	for (int i = a.x, j = b.x; i < a.y && j < b.y; i++, j++) {
		char d = lowercase(heap[i]) - lowercase(heap[j]);
		if (d != 0) return d;
	}
	return strLen(a) - strLen(b);
}

string concat(string a, string b) {
	string c = malloc(strLen(a) + strLen(b));
	strCopy(c, a);
	strCopy(c + ivec2(strLen(a), 0), b);
	return c;
}

int indexOf(string s, char c) {
	for (int i = s.x; i < s.y; i++) {
		if (heap[i] == c) return i - s.x;
	}
	return -1;
}

int indexOfI(string s, char c) {
	int lc = lowercase(c);
	for (int i = s.x; i < s.y; i++) {
		if (lowercase(heap[i]) == lc) return i - s.x;
	}
	return -1;
}

int indexOf(string s, string key) {
	int len = strLen(key);
	if (len == 0) return 0;
	if (len < 0) return -1;
	for (int i = s.x; i <= s.y-len; i++) {
		if (heap[i] == heap[key.x]) {
			if (strCmp(string(i, i+len), key) == 0) {
				return i - s.x;
			}
		}
	}
	return -1;
}

int indexOfI(string s, string key) {
	int len = strLen(key);
	if (len == 0) return 0;
	if (len < 0) return -1;
	for (int i = s.x; i <= s.y-len; i++) {
		if (lowercase(heap[i]) == lowercase(heap[key.x])) {
			if (strCmpI(string(i, i+len), key) == 0) {
				return i - s.x;
			}
		}
	}
	return -1;
}

int lastIndexOf(string s, char c) {
	for (int i = s.y-1; i >= s.x; i--) {
		if (heap[i] == c) return i - s.x;
	}
	return -1;
}

int lastIndexOf(string s, string key) {
	int len = strLen(key);
	if (len == 0) return strLen(s);
	if (len < 0) return -1;
	for (int i = s.y-len; i >= s.x; i--) {
		if (strCmp(string(i, i+len), key) == 0) {
			return i - s.x;
		}
	}
	return -1;
}

int lastIndexOfI(string s, char c) {
	int lc = lowercase(c);
	for (int i = s.y-1; i >= s.x; i--) {
		if (lowercase(heap[i]) == lc) return i - s.x;
	}
	return -1;
}

int lastIndexOfI(string s, string key) {
	int len = strLen(key);
	if (len == 0) return strLen(s);
	if (len < 0) return -1;
	for (int i = s.y-len; i >= s.x; i--) {
		if (strCmpI(string(i, i+len), key) == 0) {
			return i - s.x;
		}
	}
	return -1;
}

int normalizeIndex(int i, int len) {
	return clamp((i < 0) ? i + len : i, 0, len - 1);
}

string slice(string s, int start, int end) {
	int len = strLen(s);
	start = normalizeIndex(start, len);
	end = normalizeIndex(end, len);
	return string(s.x + start, s.x + end);
}

string slice(string s, int start) {
	int len = strLen(s);
	start = normalizeIndex(start, len);
	return string(s.x + start, s.y);
}

ivec4 splitOnce(string s, char separator) {
	int idx = indexOf(s, separator);
	ivec4 pair;
	if (idx < 0) idx = s.y - s.x;
	pair.xy = string(s.x, s.x + idx);
	pair.zw = string(s.x + idx + 1, s.y);
	return pair;
}

ivec4 splitOnce(string s, string separator) {
	int idx = indexOf(s, separator);
	ivec4 pair;
	if (idx < 0) idx = s.y - s.x;
	pair.xy = string(s.x, s.x + idx);
	pair.zw = string(s.x + idx + strLen(separator), s.y);
	return pair;
}

stringArray split(string s, char separator) {
	int start = i32heapPtr;
	string rest = s;
	while (strLen(rest) >= 0) {
		ivec4 pair = splitOnce(rest, separator);
		rest = pair.zw;
		i32heap[i32heapPtr++] = pair.x;
		i32heap[i32heapPtr++] = pair.y;
	}
	return stringArray(start, i32heapPtr);
}

stringArray split(string s, string separator) {
	int start = i32heapPtr;
	string rest = s;
	while (strLen(rest) >= 0) {
		ivec4 pair = splitOnce(rest, separator);
		rest = pair.zw;
		i32heap[i32heapPtr++] = pair.x;
		i32heap[i32heapPtr++] = pair.y;
	}
	return stringArray(start, i32heapPtr);
}

string join(stringArray a, string joiner) {
	int start = heapPtr;
	int joinerLen = strLen(joiner);
	for (int i = a.x; i < a.y-2; i += 2) {
		string str = string(i32heap[i], i32heap[i+1]);
		int len = strLen(str);
		strCopy(string(heapPtr, heapPtr+len), str);
		heapPtr += len;
		len = joinerLen;
		strCopy(string(heapPtr, heapPtr+len), joiner);
		heapPtr += len;
	}
	if (a.y > a.x) {
		string str = string(i32heap[a.y-2], i32heap[a.y-1]);
		int len = strLen(str);
		strCopy(string(heapPtr, heapPtr+len), str);
		heapPtr += len;
	}
	return string(start, heapPtr);
}

string join(stringArray a, char joiner) {
	int start = heapPtr;
	for (int i = a.x; i < a.y-2; i += 2) {
		string str = string(i32heap[i], i32heap[i+1]);
		int len = strLen(str);
		strCopy(string(heapPtr, heapPtr+len), str);
		heapPtr += len;
		heap[heapPtr] = joiner;
		heapPtr += 1;
	}
	if (a.y > a.x) {
		string str = string(i32heap[a.y-2], i32heap[a.y-1]);
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
	strCopy(s, pair.xy);
	strCopy(string(s.x + xyLen, s.y), joiner);
	strCopy(string(s.x + xyLen + joinerLen, s.y), pair.zw);
	return s;
}

string join(ivec4 pair, char joiner) {
	int zwLen = strLen(pair.zw);
	if (zwLen < 0) {
		return pair.xy;
	}
	int xyLen = strLen(pair.xy);
	int joinerLen = 1;
	string s = malloc(xyLen + joinerLen + zwLen);
	strCopy(s, pair.xy);
	heap[s.x + xyLen] = joiner;
	strCopy(string(s.x + xyLen + joinerLen, s.y), pair.zw);
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
	strCopy(moved, res);
	heapPtr = moved.y;
	return moved;
}

string repeat(char c, int count) {
	count = max(0, count);
	string s = malloc(count);
	for (int i = s.x; i < s.y; i++) {
		heap[i] = c;
	}
	return s;
}

string repeat(string pattern, int count) {
	count = max(0, count);
	int len = strLen(pattern);
	string s = malloc(count * len);
	for (int i = s.x; i < s.y; i += len) {
		strCopy(string(i, s.y), pattern);
	}
	return s;
} 

string padStart(string s, int len, char filler) {
	len = max(0, len);
	int slen = max(0, strLen(s));
	int diff = len - slen;
	string res = malloc(max(slen, len));
	for (int i = 0; i < diff; i++) {
		setC(res, i, filler);
	}
	strCopy(res + ivec2(max(0,diff), 0), s);
	return res;
}

string padEnd(string s, int len, char filler) {
	len = max(0, len);
	int slen = max(0, strLen(s));
	int diff = len - slen;
	string res = malloc(max(slen, len));
	for (int i = 0; i < diff; i++) {
		setC(res, slen+i, filler);
	}
	strCopy(res, s);
	return res;
}

string truncate(string s, int maxLen) {
	maxLen = max(0, maxLen);
	return string(s.x, s.x + min(maxLen, max(0, strLen(s))));
}

string truncateEnd(string s, int maxLen) {
	maxLen = max(0, maxLen);
	return string(s.x + max(0, max(0, strLen(s)) - maxLen), s.y);
}

bool startsWith(string s, string prefix) {
	return 0 == strCmp(truncate(s, strLen(prefix)), prefix);
}

bool endsWith(string s, string suffix) {
	return 0 == strCmp(truncateEnd(s, strLen(suffix)), suffix);
}

bool includes(string s, string key) {
	return indexOf(s, key) >= 0;
}

string trimStart(string s) {
	for (int i = s.x; i < s.y; i++) {
		char c = heap[i];
		if (isWhitespace(c)) {
			s.x++;
		} else {
			break;
		}
	}
	return s;
}

string trimEnd(string s) {
	for (int i = s.y-1; i >= s.x; i--) {
		char c = heap[i];
		if (isWhitespace(c)) {
			s.y--;
		} else {
			break;
		}
	}
	return s;
}

string trim(string s) {
	return trimEnd(trimStart(s));
}

%%GLOBALS%%

void initGlobals() {
%%INIT%%
}

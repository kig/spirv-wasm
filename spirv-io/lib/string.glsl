#include <malloc.glsl>

#define char uint8_t
#include <chr.glsl>

#define stringArray alloc_t
#define string alloc_t

#define _w(c) heap[heapPtr++] = char(c)
#define _(c) heap[heapPtr++] = CHR_##c;

layout(std430, binding = 0) buffer charBuffer { char heap[]; };

char toChar(int i) {
    return char(i);
}

size_t strLen(string str) {
    return str.y - str.x;
}

size_t arrLen(stringArray arr) {
    return (arr.y - arr.x) / 2;
}

string aGet(stringArray arr, size_t index) {
    if (index >= arrLen(arr)) return string(0, 0);
    return string(indexHeap[arr.x + index * 2], indexHeap[arr.x + index * 2 + 1]);
}

bool aSet(stringArray arr, size_t index, string value) {
    if (index >= arrLen(arr)) return false;
    indexHeap[arr.x + index * 2] = value.x;
    indexHeap[arr.x + index * 2 + 1] = value.y;
    return true;
}

string last(stringArray arr) {
    return aGet(arr, arrLen(arr)-1);
}

string first(stringArray arr) {
    return aGet(arr, 0);
}

char getC(string s, size_t index) {
    return heap[s.x + index];
}

void setC(string s, size_t index, char value) {
    heap[s.x + index] = value;
}

void strCopy(string dst, string src) {
    for (ptr_t i = dst.x, j = src.x; i < dst.y && j < src.y; i++, j++) {
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
    return c + char((c >= CHR_A && c <= CHR_Z) ? (CHR_a - CHR_A) : char(0));
}

char uppercase(char c) {
    return c + char((c >= CHR_a && c <= CHR_z) ? (CHR_A - CHR_a) : char(0));
}

string lowercaseInPlace(string s) {
    for (ptr_t i = s.x; i < s.y; i++) {
        heap[i] = lowercase(heap[i]);
    }
    return s;
}

string uppercaseInPlace(string s) {
    for (ptr_t i = s.x; i < s.y; i++) {
        heap[i] = uppercase(heap[i]);
    }
    return s;
}

string capitalizeInPlace(string s) {
    bool afterSpace = true;
    for (ptr_t i = s.x; i < s.y; i++) {
        char c = heap[i];
        c += char((afterSpace && c >= CHR_a && c <= CHR_z) ? (CHR_A - CHR_a) : char(0));
        heap[i] = c;
        afterSpace = (c == ' ' || c == '\t' || c == '\r' || c == '\n');
    }
    return s;
}

string reverseInPlace(string s) {
    for (ptr_t i = s.x, j = s.y-1; i < j; i++, j--) {
        char tmp = heap[i];
        heap[i] = heap[j];
        heap[j] = tmp;
    }
    return s;
}

string lowercase(string s) {
    return lowercaseInPlace(clone(s));
}

string uppercase(string s) {
    return uppercaseInPlace(clone(s));
}

string capitalize(string s) {
    return capitalizeInPlace(clone(s));
}

string reverse(string s) {
    return reverseInPlace(clone(s));
}

int strCmp(string a, string b) {
    for (ptr_t i = a.x, j = b.x; i < a.y && j < b.y; i++, j++) {
        int d = int(heap[i]) - int(heap[j]);
        if (d != 0) return d;
    }
    size_t al = strLen(a);
    size_t bl = strLen(b);
    return al < bl ? -1 : ((al > bl) ? 1 : 0) ;
}

int strCmpI(string a, string b) {
    for (ptr_t i = a.x, j = b.x; i < a.y && j < b.y; i++, j++) {
        int d = int(lowercase(heap[i])) - int(lowercase(heap[j]));
        if (d != 0) return d;
    }
    size_t al = strLen(a);
    size_t bl = strLen(b);
    return al < bl ? -1 : ((al > bl) ? 1 : 0) ;
}

bool strEq(string a, string b) {
    return strCmp(a, b) == 0;
}

bool strEqI(string a, string b) {
    return strCmpI(a, b) == 0;
}

string concat(string a) {
    return a;
}

string concat(string a, string b) {
    string c = malloc(strLen(a) + strLen(b));
    strCopy(c, a);
    strCopy(string(c.x + strLen(a), c.y), b);
    return c;
}

string concat(string a, string b, string c) {
    string r = malloc(strLen(a) + strLen(b) + strLen(c));
    strCopy(r, a);
    strCopy(string(r.x + strLen(a), r.y), b);
    strCopy(string(r.x + strLen(a)+strLen(b), r.y), c);
    return r;
}

string concat(string a, string b, string c, string d) {
    string r = malloc(strLen(a) + strLen(b) + strLen(c) + strLen(d));
    strCopy(r, a);
    strCopy(string(r.x + strLen(a), r.y), b);
    strCopy(string(r.x + strLen(a)+strLen(b), r.y), c);
    strCopy(string(r.x + strLen(a)+strLen(b)+strLen(c), r.y), d);
    return r;
}

string concat(string a, string b, string c, string d, string e) {
    string r = malloc(strLen(a) + strLen(b) + strLen(c) + strLen(d) + strLen(e));
    strCopy(r, a);
    strCopy(string(r.x + strLen(a), r.y), b);
    strCopy(string(r.x + strLen(a)+strLen(b), r.y), c);
    strCopy(string(r.x + strLen(a)+strLen(b)+strLen(c), r.y), d);
    strCopy(string(r.x + strLen(a)+strLen(b)+strLen(c)+strLen(d), r.y), e);
    return r;
}

string concat(string a, string b, string c, string d, string e, string f) {
    string r = malloc(strLen(a) + strLen(b) + strLen(c) + strLen(d) + strLen(e) + strLen(f));
    strCopy(r, a);
    strCopy(string(r.x + strLen(a), r.y), b);
    strCopy(string(r.x + strLen(a)+strLen(b), r.y), c);
    strCopy(string(r.x + strLen(a)+strLen(b)+strLen(c), r.y), d);
    strCopy(string(r.x + strLen(a)+strLen(b)+strLen(c)+strLen(d), r.y), e);
    strCopy(string(r.x + strLen(a)+strLen(b)+strLen(c)+strLen(d)+strLen(e), r.y), f);
    return r;

}

string concat(string a, string b, string c, string d, string e, string f, string g) {
    string r = malloc(strLen(a) + strLen(b) + strLen(c) + strLen(d) + strLen(e) + strLen(f) + strLen(g));
    strCopy(r, a);
    strCopy(string(r.x + strLen(a), r.y), b);
    strCopy(string(r.x + strLen(a)+strLen(b), r.y), c);
    strCopy(string(r.x + strLen(a)+strLen(b)+strLen(c), r.y), d);
    strCopy(string(r.x + strLen(a)+strLen(b)+strLen(c)+strLen(d), r.y), e);
    strCopy(string(r.x + strLen(a)+strLen(b)+strLen(c)+strLen(d)+strLen(e), r.y), f);
    strCopy(string(r.x + strLen(a)+strLen(b)+strLen(c)+strLen(d)+strLen(e)+strLen(f), r.y), g);
    return r;

}

string str(uint i) {
    ptr_t start = heapPtr;
    if (i == 0) heap[heapPtr++] = CHR_0;
    while (i > 0) {
        heap[heapPtr++] = CHR_0 + char(i % 10);
        i /= 10;
    }
    return reverseInPlace(string(start, heapPtr));
}

string str(uint64_t i) {
    ptr_t start = heapPtr;
    if (i == 0) heap[heapPtr++] = CHR_0;
    while (i > 0) {
        heap[heapPtr++] = CHR_0 + char(i % 10);
        i /= 10;
    }
    return reverseInPlace(string(start, heapPtr));
}

string str(int i) {
    ptr_t start = heapPtr;
    if (i < 0) heap[heapPtr++] = CHR_DASH;
    else if (i == 0) heap[heapPtr++] = CHR_0;
    ptr_t numStart = heapPtr;
    i = abs(i);
    while (i > 0) {
        heap[heapPtr++] = CHR_0 + char(i % 10);
        i /= 10;
    }
    reverseInPlace(string(numStart, heapPtr));
    return string(start, heapPtr);
}

string str(int64_t i) {
    ptr_t start = heapPtr;
    if (i < 0) heap[heapPtr++] = CHR_DASH;
    else if (i == 0) heap[heapPtr++] = CHR_0;
    ptr_t numStart = heapPtr;
    i = abs(i);
    while (i > 0) {
        heap[heapPtr++] = CHR_0 + char(i % 10);
        i /= 10;
    }
    reverseInPlace(string(numStart, heapPtr));
    return string(start, heapPtr);
}

string str(string s) {
    return s;
}

string str(float i) {
    ptr_t start = heapPtr;
    if (i < 0.0) heap[heapPtr++] = CHR_DASH;
    else if (i == 0.0) heap[heapPtr++] = CHR_0;
    ptr_t numStart = heapPtr;
    i = abs(i);
    float fi = fract(i);
    while (i > 1.0) {
        heap[heapPtr++] = CHR_0 + char(mod(i, 10.0));
        i /= 10.0;
    }
    reverseInPlace(string(numStart, heapPtr));
    if (fi > 0.0) {
        heap[heapPtr++] = CHR_DOT;
        numStart = heapPtr;
        i = fi * 10000.0;
        while (i > 1.0) {
            heap[heapPtr++] = CHR_0 + char(mod(i, 10.0));
            i /= 10.0;
        }
        reverseInPlace(string(numStart, heapPtr));
    }
    return string(start, heapPtr);
}

string str(ivec2 v) {
    ptr_t start = heapPtr;
    _(i)_(v)_(e)_(c)_(2);
    _w('(');
    str(v.x);
    _w(',');
    str(v.y);
    _w(')');
    return string(start, heapPtr);
}

string str(ivec3 v) {
    ptr_t start = heapPtr;
    _(i)_(v)_(e)_(c)_(3);
    _w('(');
    str(v.x);
    _w(',');
    str(v.y);
    _w(',');
    str(v.z);
    _w(')');
    return string(start, heapPtr);
}

string str(ivec4 v) {
    ptr_t start = heapPtr;
    _(i)_(v)_(e)_(c)_(4);
    _w('(');
    str(v.x);
    _w(',');
    str(v.y);
    _w(',');
    str(v.z);
    _w(',');
    str(v.w);
    _w(')');
    return string(start, heapPtr);
}

string str(uvec2 v) {
    ptr_t start = heapPtr;
    _(u)_(v)_(e)_(c)_(2);
    _w('(');
    str(v.x);
    _w(',');
    str(v.y);
    _w(')');
    return string(start, heapPtr);
}

string str(uvec3 v) {
    ptr_t start = heapPtr;
    _(u)_(v)_(e)_(c)_(3);
    _w('(');
    str(v.x);
    _w(',');
    str(v.y);
    _w(',');
    str(v.z);
    _w(')');
    return string(start, heapPtr);
}

string str(uvec4 v) {
    ptr_t start = heapPtr;
    _(u)_(v)_(e)_(c)_(4);
    _w('(');
    str(v.x);
    _w(',');
    str(v.y);
    _w(',');
    str(v.z);
    _w(',');
    str(v.w);
    _w(')');
    return string(start, heapPtr);
}

string str(char c) {
    heap[heapPtr++] = c;
    return string(heapPtr-1, heapPtr);
}


size_t indexOf(string s, char c) {
    for (ptr_t i = s.x; i < s.y; i++) {
        if (heap[i] == c) return i - s.x;
    }
    return -1;
}

size_t indexOfI(string s, char c) {
    char lc = lowercase(c);
    for (ptr_t i = s.x; i < s.y; i++) {
        if (lowercase(heap[i]) == lc) return i - s.x;
    }
    return -1;
}

size_t indexOf(string s, string key) {
    size_t len = strLen(key);
    if (len == 0) return 0;
    if (len < 0) return -1;
    for (ptr_t i = s.x; i <= s.y-len; i++) {
        if (heap[i] == heap[key.x]) {
            if (strCmp(string(i, i+len), key) == 0) {
                return i - s.x;
            }
        }
    }
    return -1;
}

size_t indexOfI(string s, string key) {
    size_t len = strLen(key);
    if (len == 0) return 0;
    if (len < 0) return -1;
    for (ptr_t i = s.x; i <= s.y-len; i++) {
        if (lowercase(heap[i]) == lowercase(heap[key.x])) {
            if (strCmpI(string(i, i+len), key) == 0) {
                return i - s.x;
            }
        }
    }
    return -1;
}

size_t lastIndexOf(string s, char c) {
    for (ptr_t i = s.y-1; i >= s.x; i--) {
        if (heap[i] == c) return i - s.x;
    }
    return -1;
}

size_t lastIndexOf(string s, string key) {
    size_t len = strLen(key);
    if (len == 0) return strLen(s);
    if (len < 0) return -1;
    for (ptr_t i = s.y-len; i >= s.x; i--) {
        if (strCmp(string(i, i+len), key) == 0) {
            return i - s.x;
        }
    }
    return -1;
}

size_t lastIndexOfI(string s, char c) {
    char lc = lowercase(c);
    for (ptr_t i = s.y-1; i >= s.x; i--) {
        if (lowercase(heap[i]) == lc) return i - s.x;
    }
    return -1;
}

size_t lastIndexOfI(string s, string key) {
    size_t len = strLen(key);
    if (len == 0) return strLen(s);
    if (len < 0) return -1;
    for (ptr_t i = s.y-len; i >= s.x; i--) {
        if (strCmpI(string(i, i+len), key) == 0) {
            return i - s.x;
        }
    }
    return -1;
}

size_t normalizeIndex(size_t i, size_t len) {
    return clamp((i < 0) ? i + len : i, 0, len);
}

string slice(string s, size_t start, size_t end) {
    size_t len = strLen(s);
    start = normalizeIndex(start, len);
    end = normalizeIndex(end, len);
    return string(s.x + start, s.x + end);
}

string slice(string s, size_t start) {
    size_t len = strLen(s);
    start = normalizeIndex(start, len);
    return string(s.x + start, s.y);
}

pair_t splitOnce(string s, char separator) {
    size_t idx = indexOf(s, separator);
    pair_t pair;
    if (idx < 0) idx = s.y - s.x;
    pair.x = string(s.x, s.x + idx);
    pair.y = string(s.x + idx + 1, s.y);
    return pair;
}

pair_t splitOnce(string s, string separator) {
    size_t idx = indexOf(s, separator);
    pair_t pair;
    if (idx < 0) idx = s.y - s.x;
    pair.x = string(s.x, s.x + idx);
    pair.y = string(s.x + idx + strLen(separator), s.y);
    return pair;
}

stringArray split(string s, char separator) {
    ptr_t start = toIndexPtr(heapPtr);
    ptr_t p = start;
    string rest = s;
    while (strLen(rest) >= 0) {
        pair_t pair = splitOnce(rest, separator);
        rest = pair.y;
        indexHeap[p++] = pair.x.x;
        indexHeap[p++] = pair.x.y;
    }
    heapPtr = p * INDEX_SIZE;
    return stringArray(start, p);
}

stringArray split(string s, string separator) {
    ptr_t start = toIndexPtr(heapPtr);
    ptr_t p = start;
    string rest = s;
    while (strLen(rest) >= 0) {
        pair_t pair = splitOnce(rest, separator);
        rest = pair.y;
        indexHeap[p++] = pair.x.x;
        indexHeap[p++] = pair.x.y;
    }
    heapPtr = p * INDEX_SIZE;
    return stringArray(start, p);
}

string join(stringArray a, string joiner) {
    ptr_t start = heapPtr;
    size_t joinerLen = strLen(joiner);
    for (ptr_t i = a.x; i < a.y-2; i += 2) {
        string str = string(indexHeap[i], indexHeap[i+1]);
        size_t len = strLen(str);
        strCopy(string(heapPtr, heapPtr+len), str);
        heapPtr += len;
        len = joinerLen;
        strCopy(string(heapPtr, heapPtr+len), joiner);
        heapPtr += len;
    }
    if (a.y > a.x) {
        string str = string(indexHeap[a.y-2], indexHeap[a.y-1]);
        size_t len = strLen(str);
        strCopy(string(heapPtr, heapPtr+len), str);
        heapPtr += len;
    }
    return string(start, heapPtr);
}

string join(stringArray a, char joiner) {
    ptr_t start = heapPtr;
    for (ptr_t i = a.x; i < a.y-2; i += 2) {
        string str = string(indexHeap[i], indexHeap[i+1]);
        size_t len = strLen(str);
        strCopy(string(heapPtr, heapPtr+len), str);
        heapPtr += len;
        heap[heapPtr] = joiner;
        heapPtr += 1;
    }
    if (a.y > a.x) {
        string str = string(indexHeap[a.y-2], indexHeap[a.y-1]);
        size_t len = strLen(str);
        strCopy(string(heapPtr, heapPtr+len), str);
        heapPtr += len;
    }
    return string(start, heapPtr);
}

string join(pair_t pair, string joiner) {
    size_t zwLen = strLen(pair.y);
    if (zwLen < 0) {
        return pair.x;
    }
    size_t xyLen = strLen(pair.x);
    size_t joinerLen = strLen(joiner);
    string s = malloc(xyLen + joinerLen + zwLen);
    strCopy(s, pair.x);
    strCopy(string(s.x + xyLen, s.y), joiner);
    strCopy(string(s.x + xyLen + joinerLen, s.y), pair.y);
    return s;
}

string join(pair_t pair, char joiner) {
    size_t zwLen = strLen(pair.y);
    if (zwLen < 0) {
        return pair.x;
    }
    size_t xyLen = strLen(pair.x);
    size_t joinerLen = 1;
    string s = malloc(xyLen + joinerLen + zwLen);
    strCopy(s, pair.x);
    heap[s.x + xyLen] = joiner;
    strCopy(string(s.x + xyLen + joinerLen, s.y), pair.y);
    return s;
}

string replaceOnce(string s, string pattern, string replacement) {
    ptr_t ptr = heapPtr;
    pair_t pair = splitOnce(s, pattern);
    return join(pair, replacement);
}

string replace(string s, string pattern, string replacement) {
    ptr_t ptr = heapPtr;
    stringArray a = split(s, pattern);
    string res = join(a, replacement);

    string moved = string(ptr, ptr + strLen(res));
    strCopy(moved, res);
    heapPtr = moved.y;
    return moved;
}

string repeat(char c, size_t count) {
    count = max(0, count);
    string s = malloc(count);
    for (ptr_t i = s.x; i < s.y; i++) {
        heap[i] = c;
    }
    return s;
}

string repeat(string pattern, size_t count) {
    count = max(0, count);
    size_t len = strLen(pattern);
    string s = malloc(count * len);
    for (ptr_t i = s.x; i < s.y; i += len) {
        strCopy(string(i, s.y), pattern);
    }
    return s;
}

string padStart(string s, size_t len, char filler) {
    len = max(0, len);
    size_t slen = max(0, strLen(s));
    size_t diff = len - slen;
    string res = malloc(max(slen, len));
    for (ptr_t i = 0; i < diff; i++) {
        setC(res, i, filler);
    }
    strCopy(string(res.x + max(0,diff), res.y), s);
    return res;
}

string padEnd(string s, size_t len, char filler) {
    len = max(0, len);
    size_t slen = max(0, strLen(s));
    size_t diff = len - slen;
    string res = malloc(max(slen, len));
    for (ptr_t i = 0; i < diff; i++) {
        setC(res, slen+i, filler);
    }
    strCopy(res, s);
    return res;
}

string truncate(string s, size_t maxLen) {
    maxLen = max(0, maxLen);
    return string(s.x, s.x + min(maxLen, max(0, strLen(s))));
}

string truncateEnd(string s, size_t maxLen) {
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
    for (ptr_t i = s.x; i < s.y; i++) {
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
    for (ptr_t i = s.y-1; i >= s.x; i--) {
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

int32_t parsei32(string s) {
    int32_t v = 0;
    int32_t f = 1;
    int32_t ex = 1;
    for (ptr_t i = s.y-1; i >= s.x; --i) {
        char c = heap[i];
        if (c == '-') {
            f = -1;
        } else if (c >= '0' && c <= '9') {
            v += ex * int32_t(c - '0');
            ex *= 10;
        }
    }
    return f * v;
}

float parsef32(string s) {
    float v = 0.0;
    float f = 1.0;
    float ex = 1.0;
    for (ptr_t i = s.y-1; i >= s.x; --i) {
        char c = heap[i];
        if (c == '-') {
            f = -1.0;
        } else if (c >= '0' && c <= '9') {
            v += ex * float(c - '0');
            ex *= 10.0;
        } else if (c == '.') {
            v /= ex;
            ex = 1.0;
        } else if (c == 'e' || c == 'E') {
            ex = pow(10.0, f * v);
            v = 0.0;
            f = 1.0;
        }
    }
    return f * v;
}

%%GLOBALS%%

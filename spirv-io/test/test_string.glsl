HeapSize = 4096;
FromIOSize = 10;
ToIOSize = 4096;

ThreadGroupCount = 1;
ThreadLocalCount = 1;

layout(std430, binding = 0) buffer outputBuffer { int32_t outputs[]; };

#include "../lib/file.glsl"



void main() {

    ptr_t op = (ThreadId+1) * (HeapSize/4) - 256;
    ptr_t start = op;

    ptr_t heapTop = heapPtr;

    string emptys = "";

    string s = malloc(10);
    size_t len = strLen(s);
    for (size_t i = 0; i < len; i++) {
        setC(s, i, CHR_A + char(i));
    }
    string t = malloc(10);
    for (size_t i = 0; i < len; i++) {
        setC(t, i, lowercase(getC(s, i)));
    }
    string c = concat(s, t);

    outputs[op++] = strCmp(c, "ABCDEFGHIJabcdefghij") == 0 ? 1 : -1; // 0
    outputs[op++] = indexOf(c, "X") == -1 ? 1 : -1; // 1
    outputs[op++] = indexOf(c, "A") == 0 ? 1 : -1; // 2
    outputs[op++] = indexOf(c, "a") == 10 ? 1 : -1; // 3
    outputs[op++] = indexOfI(c, "x") == -1 ? 1 : -1; // 4
    outputs[op++] = indexOfI(c, "a") == 0 ? 1 : -1; // 5

    outputs[op++] = indexOf(c, "EFG") == 4 ? 1 : -1;
    outputs[op++] = indexOf(c, "EFH") == -1 ? 1 : -1;
    outputs[op++] = indexOf(c, "ABCD") == 0 ? 1 : -1;
    outputs[op++] = indexOf(c, "hij") == 17 ? 1 : -1;
    outputs[op++] = indexOf(c, "ij") == 18 ? 1 : -1; // 10

    outputs[op++] = indexOf(c, "j") == 19 ? 1 : -1;
    outputs[op++] = indexOf(c, emptys) == 0 ? 1 : -1;

    outputs[op++] = indexOf(c, 'X') == -1 ? 1 : -1;
    outputs[op++] = indexOf(c, 'A') == 0 ? 1 : -1;
    outputs[op++] = indexOf(c, 'a') == 10 ? 1 : -1; // 15

    outputs[op++] = indexOfI(c, 'x') == -1 ? 1 : -1;
    outputs[op++] = indexOfI(c, 'a') == 0 ? 1 : -1;
    outputs[op++] = indexOf(c, 'j') == 19 ? 1 : -1;

    string csv = ",a,b,,xxx,z,,";
    {
    pair_t pair = splitOnce(csv, ",");
    outputs[op++] = strLen(pair.x) == 0 ? 1 : -1;
    outputs[op++] = strCmp(pair.y, slice(csv, 1)) == 0 ? 1 : -1; // 20
    }
    {
    pair_t pair = splitOnce(csv, "z,,");
    outputs[op++] = strLen(pair.y) == 0 ? 1 : -1;
    outputs[op++] = strCmp(pair.x, ",a,b,,xxx,") == 0 ? 1 : -1; // 22
    }
    {
    pair_t pair = splitOnce(csv, ",,,");
    outputs[op++] = strLen(pair.y) < 0 ? 1 : -1;
    outputs[op++] = strCmp(pair.x, csv) == 0 ? 1 : -1; // 24
    }
    {
    pair_t pair = splitOnce(csv, '.');
    outputs[op++] = strLen(pair.y) < 0 ? 1 : -1;
    outputs[op++] = strCmp(pair.x, csv) == 0 ? 1 : -1; // 26
    }
    {
    pair_t pair = splitOnce(csv, ',');
    outputs[op++] = strLen(pair.x) == 0 ? 1 : -1;
    outputs[op++] = strCmp(pair.y, slice(csv, 1)) == 0 ? 1 : -1;
    outputs[op++] = strCmp(pair.y, "a,b,,xxx,z,,") == 0 ? 1 : -1; // 29
    }

    stringArray s0 = split(csv, ',');
    stringArray s1 = split(csv, ",");
    stringArray s2 = split(csv, ",,");
    string csv0 = join(s0, ',');
    string csv1 = join(s0, ",");
    string csv2 = join(s1, "woo");
    string csv3 = join(s2, "bar");
    outputs[op++] =
    strCmp(csv0, csv) == 0 ? 1 : -1;
    outputs[op++] =
    strCmp(csv1, csv) == 0 ? 1 : -1;
    outputs[op++] =
    strCmp(csv2, "wooawoobwoowooxxxwoozwoowoo") == 0 ? 1 : -1;
    outputs[op++] =
    strCmp(csv3, ",a,bbarxxx,zbar") == 0 ? 1 : -1; // 33


    outputs[op++] =
    strCmp(join(splitOnce("a,b", ','), ','), "a,b") == 0 ? 1 : -1;
    outputs[op++] =
    strCmp(join(splitOnce("ab", ','), ','), "ab") == 0 ? 1 : -1;
    outputs[op++] =
    strCmp(join(splitOnce(",b", ','), ','), ",b") == 0 ? 1 : -1;
    outputs[op++] =
    strCmp(join(splitOnce("a,", ','), ','), "a,") == 0 ? 1 : -1; // 37

    bool b0 = startsWith(c, s);
    bool b1 = startsWith(c, t);
    bool b2 = endsWith(c, t);
    bool b3 = endsWith(c, s);
    outputs[op++] = b0 ? 1 : -1; // 38
    outputs[op++] = !b1 ? 1 : -1;
    outputs[op++] = b2 ? 1 : -1; // 40
    outputs[op++] = !b3 ? 1 : -1;
    outputs[op++] = startsWith(c, c) ? 1 : -1;
    outputs[op++] = startsWith(t, t) ? 1 : -1;
    outputs[op++] = startsWith(s, s) ? 1 : -1;
    outputs[op++] = startsWith(emptys, emptys) ? 1 : -1; // 45
    outputs[op++] = startsWith(s, emptys) ? 1 : -1;
    outputs[op++] = endsWith(s, s) ? 1 : -1;
    outputs[op++] = endsWith(c, c) ? 1 : -1;
    outputs[op++] = endsWith(emptys, emptys) ? 1 : -1;
    outputs[op++] = endsWith(s, emptys) ? 1 : -1; // 50
    outputs[op++] = includes(s, s) ? 1 : -1;
    outputs[op++] = includes(c, c) ? 1 : -1;
    outputs[op++] = includes(emptys, emptys) ? 1 : -1;
    outputs[op++] = includes(s, emptys) ? 1 : -1; // 54

    outputs[op++] =
    strCmp(capitalize("hello there Bob a"), "Hello There Bob A") == 0 ? 1 : -1;
    // 55
    outputs[op++] =
    strCmp(reverse("abc"), "cba") == 0 ? 1 : -1;
    outputs[op++] =
    strCmp(reverse("ab"), "ba") == 0 ? 1 : -1;
    outputs[op++] =
    strCmp(reverse("a"), "a") == 0 ? 1 : -1; // 58

    string r0 = replace("<h1>Hi</h1>", "h1", "span");
    outputs[op++] =
    strCmp(r0, "<span>Hi</span>") == 0 ? 1 : -1; // 59

    string trims = " \t\nhi  \r\n  \t  ";
    outputs[op++] = strCmp(trimStart(trims), slice(trims, 3)) == 0 ? 1 : -1; // 60
    outputs[op++] = strCmp(trimEnd(trims), slice(trims, 0, 5)) == 0 ? 1 : -1;
    outputs[op++] = strCmp(trim(trims), slice(trims, 3, 5)) == 0 ? 1 : -1;
    outputs[op++] = strCmp(trim(slice(trims, 0, 0)), slice(trims, 0, 0)) == 0 ? 1 : -1;
    outputs[op++] = strCmp(trim(slice(trims, 0, 3)), slice(trims, 0, 0)) == 0 ? 1 : -1; // 64

    // repeat
    outputs[op++] = strCmp(repeat('a', 5), "aaaaa") == 0 ? 1 : -1; // 65
    outputs[op++] = strCmp(repeat('a', 0), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(repeat('a', 1), "a") == 0 ? 1 : -1;
    outputs[op++] = strCmp(repeat('a', -1), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(repeat("a", 5), "aaaaa") == 0 ? 1 : -1;
    outputs[op++] = strCmp(repeat("a", 0), emptys) == 0 ? 1 : -1; // 70
    outputs[op++] = strCmp(repeat("a", 1), "a") == 0 ? 1 : -1;
    outputs[op++] = strCmp(repeat("ax", 5), "axaxaxaxax") == 0 ? 1 : -1;
    outputs[op++] = strCmp(repeat("ax", 1), "ax") == 0 ? 1 : -1;
    outputs[op++] = strCmp(repeat("ax", 0), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(repeat("ax", -1), emptys) == 0 ? 1 : -1; // 75
    outputs[op++] = strCmp(repeat(emptys, 42), emptys) == 0 ? 1 : -1; // 76

    // replaceOnce
    string replaceOnces = "jam ham dam yam";
    outputs[op++] = strCmp(replaceOnce(replaceOnces, "am", "oost"), "joost ham dam yam") == 0 ? 1 : -1; // 77
    outputs[op++] = strCmp(replaceOnce(replaceOnces, emptys, "oost"), "oostjam ham dam yam") == 0 ? 1 : -1;
    outputs[op++] = strCmp(replaceOnce(replaceOnces, "am", emptys), "j ham dam yam") == 0 ? 1 : -1;
    outputs[op++] = strCmp(replaceOnce(replaceOnces, "afm", "oost"), replaceOnces) == 0 ? 1 : -1; // 80
    outputs[op++] = strCmp(replaceOnce(emptys, "afm", "oost"), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(replaceOnce(emptys, emptys, emptys), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(replaceOnce(emptys, emptys, "what"), "what") == 0 ? 1 : -1; // 83

    // padStart
    outputs[op++] = strCmp(padStart("4", 4, '0'), "0004") == 0 ? 1 : -1; // 84
    outputs[op++] = strCmp(padStart(emptys, 4, '0'), "0000") == 0 ? 1 : -1; // 85
    outputs[op++] = strCmp(padStart("4", 0, '0'), "4") == 0 ? 1 : -1;
    outputs[op++] = strCmp(padStart("4", 1, '0'), "4") == 0 ? 1 : -1;
    outputs[op++] = strCmp(padStart("4", 2, '0'), "04") == 0 ? 1 : -1;
    outputs[op++] = strCmp(padStart("42", 4, '0'), "0042") == 0 ? 1 : -1;
    outputs[op++] = strCmp(padStart("42", 1, '0'), "42") == 0 ? 1 : -1; // 90
    outputs[op++] = strCmp(padStart("42", -1485, '0'), "42") == 0 ? 1 : -1; // 91

    // padEnd
    outputs[op++] = strCmp(padEnd("4", 4, '0'), "4000") == 0 ? 1 : -1; // 92
    outputs[op++] = strCmp(padEnd(emptys, 4, '0'), "0000") == 0 ? 1 : -1;
    outputs[op++] = strCmp(padEnd("4", 0, '0'), "4") == 0 ? 1 : -1;
    outputs[op++] = strCmp(padEnd("4", 1, '0'), "4") == 0 ? 1 : -1; // 95
    outputs[op++] = strCmp(padEnd("4", 2, '0'), "40") == 0 ? 1 : -1;
    outputs[op++] = strCmp(padEnd("42", 4, '0'), "4200") == 0 ? 1 : -1;
    outputs[op++] = strCmp(padEnd("42", 1, '0'), "42") == 0 ? 1 : -1;
    outputs[op++] = strCmp(padEnd("42", -1485, '0'), "42") == 0 ? 1 : -1; // 99

    string foos = "foo";

    // clone
    outputs[op++] = strCmp(clone(foos), foos) == 0 ? 1 : -1; // 100
    outputs[op++] = strCmp(clone(emptys), emptys) == 0 ? 1 : -1;

    // truncate
    outputs[op++] = strCmp(truncate(foos, 2), "fo") == 0 ? 1 : -1; // 102
    outputs[op++] = strCmp(truncate(foos, 3), foos) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncate(foos, 4), foos) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncate(foos, 1), "f") == 0 ? 1 : -1; // 105
    outputs[op++] = strCmp(truncate(foos, 0), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncate(foos, -1), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncate(emptys, 2), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncate(emptys, -2), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncate(foos, -2), emptys) == 0 ? 1 : -1; // 110

    // truncateEnd
    outputs[op++] = strCmp(truncateEnd(foos, 2), "oo") == 0 ? 1 : -1; // 111
    outputs[op++] = strCmp(truncateEnd(foos, 3), foos) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncateEnd(foos, 4), foos) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncateEnd(foos, 1), "o") == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncateEnd(foos, 0), emptys) == 0 ? 1 : -1; // 115
    outputs[op++] = strCmp(truncateEnd(foos, -1), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncateEnd(emptys, 2), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncateEnd(emptys, -2), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(truncateEnd(foos, -2), emptys) == 0 ? 1 : -1; // 119

    // uppercase
    outputs[op++] = uppercase('c') == 'C' ? 1 : -1; // 120
    outputs[op++] = uppercase('C') == 'C' ? 1 : -1;
    outputs[op++] = uppercase('a') == 'A' ? 1 : -1;
    outputs[op++] = uppercase('z') == 'Z' ? 1 : -1;
    outputs[op++] = uppercase('A') == 'A' ? 1 : -1;
    outputs[op++] = uppercase('Z') == 'Z' ? 1 : -1; // 125
    outputs[op++] = uppercase('5') == '5' ? 1 : -1;
    outputs[op++] = uppercase('\t') == '\t' ? 1 : -1;
    outputs[op++] = strCmp(uppercase(""), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(uppercase("AxEs-123 !"), "AXES-123 !") == 0 ? 1 : -1; // 129

    // lowercase
    outputs[op++] = lowercase('c') == 'c' ? 1 : -1; // 130
    outputs[op++] = lowercase('C') == 'c' ? 1 : -1;
    outputs[op++] = lowercase('a') == 'a' ? 1 : -1;
    outputs[op++] = lowercase('z') == 'z' ? 1 : -1;
    outputs[op++] = lowercase('A') == 'a' ? 1 : -1;
    outputs[op++] = lowercase('Z') == 'z' ? 1 : -1; // 135
    outputs[op++] = lowercase('5') == '5' ? 1 : -1;
    outputs[op++] = lowercase('\t') == '\t' ? 1 : -1;
    outputs[op++] = strCmp(lowercase(""), emptys) == 0 ? 1 : -1;
    outputs[op++] = strCmp(lowercase("AxEs-123 !"), "axes-123 !") == 0 ? 1 : -1; // 139

    // capitalize
    outputs[op++] = strCmp(capitalize(""), emptys) == 0 ? 1 : -1; // 140
    outputs[op++] = strCmp(capitalize("jabba wabba"), "Jabba Wabba") == 0 ? 1 : -1;
    outputs[op++] = strCmp(capitalize(" ark\nb.\tnic\rx.y.z"), " Ark\nB.\tNic\rX.y.z") == 0 ? 1 : -1; // 142

    // isWhitespace
    outputs[op++] = !isWhitespace('z') ? 1 : -1; // 143
    outputs[op++] = !isWhitespace('A') ? 1 : -1;
    outputs[op++] = !isWhitespace('A') ? 1 : -1; // 145
    outputs[op++] = !isWhitespace('5') ? 1 : -1;
    outputs[op++] = isWhitespace('\t') ? 1 : -1;
    outputs[op++] = isWhitespace('\n') ? 1 : -1;
    outputs[op++] = isWhitespace(' ') ? 1 : -1;
    outputs[op++] = isWhitespace('\r') ? 1 : -1; // 150

    // lastIndexOf
    outputs[op++] = lastIndexOf(foos, 'o') == 2 ? 1 : -1; // 151
    outputs[op++] = lastIndexOf(foos, 'f') == 0 ? 1 : -1;
    outputs[op++] = lastIndexOf(foos, 'x') == -1 ? 1 : -1;
    outputs[op++] = lastIndexOf(foos, "o") == 2 ? 1 : -1;
    outputs[op++] = lastIndexOf(foos, "f") == 0 ? 1 : -1; // 155
    outputs[op++] = lastIndexOf(foos, "x") == -1 ? 1 : -1;
    outputs[op++] = lastIndexOf(foos, "oo") == 1 ? 1 : -1;
    outputs[op++] = lastIndexOf(foos, "fo") == 0 ? 1 : -1;
    outputs[op++] = lastIndexOf(foos, "foo") == 0 ? 1 : -1;
    outputs[op++] = lastIndexOf(foos, "foos") == -1 ? 1 : -1; // 160
    outputs[op++] = lastIndexOf(foos, "") == 3 ? 1 : -1; // 161

    // lastIndexOfI
    outputs[op++] = lastIndexOfI(foos, 'o') == 2 ? 1 : -1; // 162
    outputs[op++] = lastIndexOfI(foos, 'F') == 0 ? 1 : -1;
    outputs[op++] = lastIndexOfI(foos, 'x') == -1 ? 1 : -1;
    outputs[op++] = lastIndexOfI(foos, "O") == 2 ? 1 : -1; // 165
    outputs[op++] = lastIndexOfI(foos, "f") == 0 ? 1 : -1;
    outputs[op++] = lastIndexOfI(foos, "X") == -1 ? 1 : -1;
    outputs[op++] = lastIndexOfI(foos, "Oo") == 1 ? 1 : -1;
    outputs[op++] = lastIndexOfI(foos, "Fo") == 0 ? 1 : -1;
    outputs[op++] = lastIndexOfI(foos, "fOo") == 0 ? 1 : -1; // 170
    outputs[op++] = lastIndexOfI(foos, "fOoS") == -1 ? 1 : -1;
    outputs[op++] = lastIndexOfI(foos, "") == 3 ? 1 : -1; // 172

    // strCmp
    outputs[op++] = strCmp("", "") == 0 ? 1 : -1; // 173
    outputs[op++] = strCmp("a", "b") < 0 ? 1 : -1;
    outputs[op++] = strCmp("b", "a") > 0 ? 1 : -1; // 175
    outputs[op++] = strCmp("B", "a") < 0 ? 1 : -1;
    outputs[op++] = strCmp("123", "1234") < 0 ? 1 : -1;
    outputs[op++] = strCmp("1234", "123") > 0 ? 1 : -1;
    outputs[op++] = strCmp("", "123") < 0 ? 1 : -1;
    outputs[op++] = strCmp("123", "") > 0 ? 1 : -1; // 180

    // strCmpI
    outputs[op++] = strCmpI("", "") == 0 ? 1 : -1; // 181
    outputs[op++] = strCmpI("a", "b") < 0 ? 1 : -1;
    outputs[op++] = strCmpI("A", "a") == 0 ? 1 : -1;
    outputs[op++] = strCmpI("b", "B") == 0 ? 1 : -1;
    outputs[op++] = strCmpI("b", "a") > 0 ? 1 : -1; // 185
    outputs[op++] = strCmpI("B", "a") > 0 ? 1 : -1;
    outputs[op++] = strCmpI("abC", "AbCd") < 0 ? 1 : -1;
    outputs[op++] = strCmpI("AbCd", "abC") > 0 ? 1 : -1;
    outputs[op++] = strCmpI("aBcD", "AbCd") == 0 ? 1 : -1;
    outputs[op++] = strCmpI("", "aBc") < 0 ? 1 : -1; // 190
    outputs[op++] = strCmpI("aBc", "") > 0 ? 1 : -1; // 191

    heapPtr = heapTop;

    bool failed = false;
    for (int i = start; i < op; i++) {
        if (outputs[i] != 1) {
            failed = true;
            FREE_ALL(
                log(concat("Error in test ", str(i), ": ", str(outputs[i])));
            )
        }
    }
    if (!failed) {
        log("All tests successful!");
    }
    println("Hello from thread ", ThreadId, ", the time now is: ", clockRealtimeEXT());
}

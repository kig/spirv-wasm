#define version #version
version 450

#define HEAP_SIZE 8192

layout ( local_size_x = 16, local_size_y = 1, local_size_z = 1 ) in;

layout(std430, binding = 0) readonly buffer inputBuffer { highp int inputs[]; };
layout(std430, binding = 1) buffer outputBuffer { highp int outputs[]; };
layout(std430, binding = 2) buffer heapBuffer { lowp int heap[]; };

#include "string.glsl"


void main() {
	initGlobals();
	
	int op = 0; 

	string s = malloc(10);
	int len = strLen(s);
	for (int i = 0; i < len; i++) {
		setC(s, i, CHR_A + i);
	}
	string t = malloc(10);
	for (int i = 0; i < len; i++) {
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
	outputs[op++] = indexOf(c, "") == 0 ? 1 : -1;

	outputs[op++] = indexOf(c, 'X') == -1 ? 1 : -1;
	outputs[op++] = indexOf(c, 'A') == 0 ? 1 : -1;
	outputs[op++] = indexOf(c, 'a') == 10 ? 1 : -1; // 15

	outputs[op++] = indexOfI(c, 'x') == -1 ? 1 : -1;
	outputs[op++] = indexOfI(c, 'a') == 0 ? 1 : -1;
	outputs[op++] = indexOf(c, 'j') == 19 ? 1 : -1;
	
	string csv = ",a,b,,xxx,z,,";
	{
	ivec4 pair = splitOnce(csv, ",");
	outputs[op++] = strLen(pair.xy) == 0 ? 1 : -1;
	outputs[op++] = strCmp(pair.zw, slice(csv, 1)) == 0 ? 1 : -1; // 20
	}
	{
	ivec4 pair = splitOnce(csv, "z,,");
	outputs[op++] = strLen(pair.zw) == 0 ? 1 : -1;
	outputs[op++] = strCmp(pair.xy, ",a,b,,xxx,") == 0 ? 1 : -1; // 22
	}
	{
	ivec4 pair = splitOnce(csv, ",,,");
	outputs[op++] = strLen(pair.zw) < 0 ? 1 : -1;
	outputs[op++] = strCmp(pair.xy, csv) == 0 ? 1 : -1; // 24
	}
	{
	ivec4 pair = splitOnce(csv, '.');
	outputs[op++] = strLen(pair.zw) < 0 ? 1 : -1;
	outputs[op++] = strCmp(pair.xy, csv) == 0 ? 1 : -1; // 26
	}
	{
	ivec4 pair = splitOnce(csv, ',');
	outputs[op++] = strLen(pair.xy) == 0 ? 1 : -1;
	outputs[op++] = strCmp(pair.zw, slice(csv, 1)) == 0 ? 1 : -1;
	outputs[op++] = strCmp(pair.zw, "a,b,,xxx,z,,") == 0 ? 1 : -1; // 29
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
	outputs[op++] = startsWith("", "") ? 1 : -1; // 45
	outputs[op++] = startsWith(s, "") ? 1 : -1;
	outputs[op++] = endsWith(s, s) ? 1 : -1;
	outputs[op++] = endsWith(c, c) ? 1 : -1;
	outputs[op++] = endsWith("", "") ? 1 : -1;
	outputs[op++] = endsWith(s, "") ? 1 : -1; // 50
	outputs[op++] = includes(s, s) ? 1 : -1;
	outputs[op++] = includes(c, c) ? 1 : -1;
	outputs[op++] = includes("", "") ? 1 : -1;
	outputs[op++] = includes(s, "") ? 1 : -1; // 54

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
	strCmp(r0, "<span>Hi</span>") == 0 ? 1 : heapPtr; // 59
}

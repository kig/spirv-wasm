#define HEAP_SIZE 8192

#include "string.glsl"

void main() {
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
	string csv = ",a,b,,xxx,z,,";
	stringArray s0 = split(csv, ',');
	stringArray s1 = split(csv, ",");
	stringArray s2 = split(csv, ",,");
	string csv0 = join(s0, ',');
	string csv1 = join(s0, ",");
	string csv2 = join(s1, "woo");
	string csv3 = join(s2, "bar");

	bool b0 = startsWith(c, s);
	bool b1 = startsWith(c, t);
	bool b2 = endsWith(c, t);
	bool b3 = endsWith(c, s);

	capitalize("hello there Bob a") == "Hello There Bob A";
	reverse("abc") == "cba";
	reverse("ab") == "ba";
	reverse("a") == "a";

	string r0 = replace("<h1>Hi</h1>", "h1", "span");
}

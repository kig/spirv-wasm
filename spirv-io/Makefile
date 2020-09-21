CPP := clang++
CFLAGS := -m64 -march=native -mtune=native -std=c++17 -I../include -O2
LDFLAGS := -ldl -llz4 -lzstd -lvulkan -lpthread

gls:
	$(CPP) $(CFLAGS) $(LDFLAGS) -o bin/gls src/gls.cpp

gls_cpu:
	$(CPP) $(CFLAGS) $(LDFLAGS) -o bin/gls_cpu src/gls_cpu.cpp

install: gls
	install -d $(DESTDIR)$(PREFIX)/lib/
	install -m 644 lib/* $(DESTDIR)$(PREFIX)/lib/
	install -d $(DESTDIR)$(PREFIX)/bin/
	install -m 755 bin/* $(DESTDIR)$(PREFIX)/bin/

all: gls

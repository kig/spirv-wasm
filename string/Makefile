CPP := clang++
CFLAGS := -m64 -O2 -march=znver1 -mtune=znver1 -std=c++17
LDFLAGS := -ldl -llz4 -lzstd -lvulkan -lpthread

gls:
	$(CPP) $(CFLAGS) $(LDFLAGS) -o bin/gls src/gls.cpp

install: gls
	install -d $(DESTDIR)$(PREFIX)/lib/
	install -m 644 lib/* $(DESTDIR)$(PREFIX)/lib/
	install -d $(DESTDIR)$(PREFIX)/bin/
	install -m 755 bin/* $(DESTDIR)$(PREFIX)/bin/

all: gls

SOURCES := $(wildcard src/*.comp)
SPIRV := $(SOURCES:.comp=.spv)
CPP_INTERFACE := $(SOURCES:.comp=.spv.cpp)
CPP_DRIVER := $(SOURCES:.comp=.cpp)
EXECUTABLES := $(SOURCES:.comp=.html)
OBJECTS := $(CPP_DRIVER:.cpp=.o) $(CPP_INTERFACE:.cpp=.o)

TOTAL_MEMORY := 67108864
TOTAL_THREADS := 16

CXXFLAGS += -std=c++11 -Iinclude -Isrc -I/usr/local/include -s WASM=1 -s USE_PTHREADS=1 -s TOTAL_MEMORY=$(TOTAL_MEMORY) -s EXTRA_EXPORTED_RUNTIME_METHODS='["ccall"]' -O3 -msimd128 -s SIMD=1
LDFLAGS += -pthread -lm -s WASM=1 -s USE_PTHREADS=1 -s TOTAL_MEMORY=$(TOTAL_MEMORY) -s PTHREAD_POOL_SIZE=$(TOTAL_THREADS) -s EXTRA_EXPORTED_RUNTIME_METHODS='["ccall"]' -O3 -msimd128 -s SIMD=1

all: $(EXECUTABLES)

%.spv: %.comp
	glslangValidator -V -o $@ $<

%.spv.cpp: %.spv
	spirv-cross --cpp --output $@ $<

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

%.html: %.o %.spv.o
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	$(RM) -f $(EXECUTABLES) $(SPIRV) $(CPP_INTERFACE) $(OBJECTS)

.PHONY: clean
.SECONDARY:


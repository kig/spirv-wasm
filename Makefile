SOURCES := $(wildcard src/*.comp)
SPIRV := $(SOURCES:.comp=.spv)
CPP_INTERFACE := $(SOURCES:.comp=.spv.cpp)
CPP_DRIVER := $(SOURCES:.comp=.cpp)
EXECUTABLES := $(SOURCES:.comp=.html)
OBJECTS := $(CPP_DRIVER:.cpp=.o) $(CPP_INTERFACE:.cpp=.o)

TOTAL_MEMORY := 67108864
TOTAL_THREADS := 16
USE_THREADS := 1
USE_SIMD := 0

ifeq ($(USE_THREADS), 1)
	THREAD_FLAGS := -pthread -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=$(TOTAL_THREADS)
endif
ifeq ($(USE_SIMD), 1)
	SIMD_FLAGS := -msimd128 -s SIMD=1
endif

CXXFLAGS += -std=c++11 -Iinclude -Isrc -I/usr/local/include -O3 -s WASM=1 $(THREAD_FLAGS) -s TOTAL_MEMORY=$(TOTAL_MEMORY) -s EXTRA_EXPORTED_RUNTIME_METHODS='["ccall"]' $(SIMD_FLAGS)
LDFLAGS += -lm -O3 -s WASM=1 $(THREAD_FLAGS) -s TOTAL_MEMORY=$(TOTAL_MEMORY) -s EXTRA_EXPORTED_RUNTIME_METHODS='["ccall"]' $(SIMD_FLAGS)

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


CXX = g++
INCLUDE = -I../rtx/external $(shell pkg-config --cflags glfw3) $(shell python3 -m pybind11 --includes) $(shell python3-config --includes)
LIBRARIES = -L/usr/local/cuda-9.1/lib64
LDFLAGS = $(shell pkg-config --static --libs glfw3) $(shell python3-config --ldflags) -pthread
CXXFLAGS = -O0 -g -Wall -Wformat -march=native -std=c++14
SOURCES = $(wildcard ../rtx/external/gl3w/*.c) \
		  $(wildcard ../rtx/core/class/*.cpp) \
		  $(wildcard ../rtx/core/geometry/*.cpp) \
		  $(wildcard ../rtx/core/material/mesh/*.cpp) \
		  $(wildcard ../rtx/core/camera/*.cpp) \
		  $(wildcard ../rtx/core/bvh/*.cpp) \
		  $(wildcard ../rtx/core/renderer/cpu/ray_tracing/*.cpp) \
		  $(wildcard ../rtx/core/renderer/cpu/*.cpp) \
		  $(wildcard ../rtx/core/renderer/cuda/ray_tracing/*.cpp) \
		  $(wildcard ../rtx/core/renderer/cuda/kernel/*.cu) \
		  $(wildcard ../rtx/core/renderer/options/*.cpp) \
		  main.cpp
OBJS = $(patsubst %.cu,%.o,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCES))))
TARGET = run

UNAME := $(shell uname -s)
ifeq ($(UNAME), Linux)
	LDFLAGS += -lGL
endif
ifeq ($(UNAME), Darwin)
	LDFLAGS += -framework OpenGL -undefined dynamic_lookup
endif

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(LIBRARIES) -lcudart 

.c.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

../rtx/core/renderer/cuda/kernel/%.o: ../rtx/core/renderer/cuda/kernel/%.cu
	nvcc -ccbin=$(CXX) -Xcompiler "-fPIC" -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)

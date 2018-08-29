CXX = g++
INCLUDE = -I./external $(shell pkg-config --cflags glfw3) $(shell python3 -m pybind11 --includes)
LIBRARIES = -L/usr/local/cuda-9.1/lib64
LDFLAGS = $(shell pkg-config --static --libs glfw3) -shared -fopenmp
CXXFLAGS = -O3 -Wall -Wformat -march=native -std=c++14 -fPIC -fopenmp
NVCCFLAGS = -ccbin=$(CXX) -Xcompiler "-fPIC" --ptxas-options=-v
SOURCES = $(wildcard ./external/gl3w/*.c) \
		  $(wildcard ./core/class/*.cpp) \
		  $(wildcard ./core/geometry/*.cpp) \
		  $(wildcard ./core/material/mesh/*.cpp) \
		  $(wildcard ./core/camera/*.cpp) \
		  $(wildcard ./core/renderer/bvh/*.cpp) \
		  $(wildcard ./core/renderer/cpu/ray_tracing/*.cpp) \
		  $(wildcard ./core/renderer/cpu/*.cpp) \
		  $(wildcard ./core/renderer/cuda/ray_tracing/*.cpp) \
		  $(wildcard ./core/renderer/cuda/kernel/*.cu) \
		  $(wildcard ./core/renderer/options/*.cpp) \
		  pybind/rtx.cpp
OBJS = $(patsubst %.cu,%.o,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCES))))
EXTENSION = $(shell python3-config --extension-suffix)
OUTPUT = ../run
TARGET = $(OUTPUT)/rtx$(EXTENSION)

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

./core/renderer/cuda/kernel/%.o: ./core/renderer/cuda/kernel/%.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)

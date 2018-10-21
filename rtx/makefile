CXX = g++
INCLUDE = -I./external $(shell pkg-config --cflags glfw3) $(shell python3 -m pybind11 --includes)
LIBRARIES = -L/usr/local/cuda-9.1/lib64
LDFLAGS = $(shell pkg-config --static --libs glfw3) -shared -fopenmp
CXXFLAGS = -O3 -Wall -Wformat -march=native -std=c++14 -fPIC -fopenmp
NVCCFLAGS = -ccbin=$(CXX) -Xcompiler "-fPIC"
SOURCES = $(wildcard ./external/gl3w/*.c) \
		  $(wildcard ./core/class/*.cpp) \
		  $(wildcard ./core/geometry/*.cpp) \
		  $(wildcard ./core/material/*.cpp) \
		  $(wildcard ./core/mapping/*.cpp) \
		  $(wildcard ./core/camera/*.cpp) \
		  $(wildcard ./core/renderer/bvh/*.cpp) \
		  $(wildcard ./core/renderer/kernel/*.cu) \
		  $(wildcard ./core/renderer/kernel/mcrt/*.cu) \
		  $(wildcard ./core/renderer/kernel/next_event_estimation/*.cu) \
		  $(wildcard ./core/renderer/*.cpp) \
		  $(wildcard ./core/renderer/arguments/*.cpp) \
		  pybind/rtx.cpp
OBJS = $(patsubst %.cu,%.o,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCES))))
EXTENSION = $(shell python3-config --extension-suffix)
OUTPUT = ../examples/python
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

./core/renderer/kernel/%.o: ./core/renderer/kernel/%.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

./core/renderer/kernel/mcrt/%.o: ./core/renderer/kernel/%.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

./core/renderer/kernel/next_event_estimation/%.o: ./core/renderer/kernel/%.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean_objects
clean_objects:
	rm -f $(OBJS) $(TARGET)

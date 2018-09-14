CXX = g++
INCLUDE = -I../rtx/external $(shell pkg-config --cflags glfw3) $(shell python3 -m pybind11 --includes) $(shell python3-config --includes)
LIBRARIES = -L/usr/local/cuda-9.1/lib64
LDFLAGS = $(shell pkg-config --static --libs glfw3) $(shell python3-config --ldflags) -pthread -fopenmp
CXXFLAGS = -O3 -Wall -Wformat -march=native -std=c++14 -fopenmp
NVCCFLAGS = -ccbin=$(CXX) --ptxas-options=-v
SOURCES = $(wildcard ../rtx/external/gl3w/*.c) \
		  $(wildcard ../rtx/core/class/*.cpp) \
		  $(wildcard ../rtx/core/geometry/*.cpp) \
		  $(wildcard ../rtx/core/material/*.cpp) \
		  $(wildcard ../rtx/core/mapping/*.cpp) \
		  $(wildcard ../rtx/core/camera/*.cpp) \
		  $(wildcard ../rtx/core/renderer/*.cpp) \
		  $(wildcard ../rtx/core/renderer/kernel/*.cu) \
		  $(wildcard ../rtx/core/renderer/bvh/*.cpp) \
		  $(wildcard ../rtx/core/renderer/arguments/*.cpp) \
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

../rtx/core/renderer/kernel/%.o: ../rtx/core/renderer/kernel/%.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)

CXX = g++
INCLUDE = -I./external $(shell pkg-config --cflags glfw3) $(shell python3 -m pybind11 --includes)
LDFLAGS = $(shell pkg-config --static --libs glfw3) -shared -pthread -fopenmp
CXXFLAGS = -O3 -DNDEBUG -Wall -Wformat -march=native -std=c++14 -fPIC -fopenmp
SOURCES = $(wildcard ./external/gl3w/*.c) \
		  $(wildcard ./core/class/*.cpp) \
		  $(wildcard ./core/geometry/*.cpp) \
		  $(wildcard ./core/material/mesh/*.cpp) \
		  $(wildcard ./core/camera/*.cpp) \
		  $(wildcard ./core/renderer/cpu/ray_tracing/*.cpp) \
		  $(wildcard ./core/renderer/cpu/*.cpp) \
		  $(wildcard ./core/renderer/options/*.cpp) \
		  pybind/rtx.cpp
OBJS = $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCES)))
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
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

.c.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)

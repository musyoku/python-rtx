CXX = g++
INCLUDE = -I../three/external $(shell pkg-config --cflags glfw3) $(shell python3 -m pybind11 --includes) $(shell python3-config --includes)
LDFLAGS = $(shell pkg-config --static --libs glfw3) $(shell python3-config --ldflags) -pthread
CXXFLAGS = -O0 -g -Wall -Wformat -march=native -std=c++14
SOURCES = $(wildcard ../three/external/gl3w/*.c) \
		  $(wildcard ../three/core/class/*.cpp) \
		  $(wildcard ../three/core/geometry/*.cpp) \
		  $(wildcard ../three/core/material/mesh/*.cpp) \
		  $(wildcard ../three/core/camera/*.cpp) \
		  $(wildcard ../three/core/renderer/cpu/ray_tracing/*.cpp) \
		  $(wildcard ../three/core/renderer/cpu/*.cpp) \
		  $(wildcard ../three/core/renderer/options/*.cpp) \
		  main.cpp
OBJS = $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCES)))
TARGET = run

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

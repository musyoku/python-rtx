#pragma once
#include "camera.h"
#include "scene.h"
#include <memory>
#include <pybind11/numpy.h>

namespace three {
class Renderer {
public:
    int _width;
    int _height;
};
}
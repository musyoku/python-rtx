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
    void render(std::shared_ptr<Scene> scene,
        std::shared_ptr<Camera> camera,
        pybind11::array_t<unsigned int, pybind11::array::c_style> buffer);
};
}
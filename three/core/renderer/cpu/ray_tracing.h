#pragma once
#include "../../class/renderer.h"
#include <pybind11/numpy.h>

namespace three {
class RayTracingCPURenderer : public Renderer {
public:
    void render(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, pybind11::array_t<int, pybind11::array::c_style> buffer);
};
}
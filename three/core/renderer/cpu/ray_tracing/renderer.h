#pragma once
#include "../../../class/renderer.h"
#include "../../options/ray_tracing.h"
#include <memory>
#include <pybind11/numpy.h>

namespace three {
class RayTracingCPURenderer : public Renderer {
public:
    void render(std::shared_ptr<Scene> scene,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<RayTracingOptions> options,
        pybind11::array_t<float, pybind11::array::c_style> buffer);
};
}
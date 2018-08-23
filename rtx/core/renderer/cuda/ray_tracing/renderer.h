#pragma once
#include "../../../class/ray.h"
#include "../../../class/renderer.h"
#include "../../options/ray_tracing.h"
#include <array>
#include <memory>
#include <pybind11/numpy.h>
#include <random>

namespace rtx {
class RayTracingCUDARenderer : public Renderer {
private:
    float* _face_vertices;
    int* _object_types;
    float* _rays;
    float* _color_per_ray;
    bool _initialized;
    float* _gpu_rays;
    float* _gpu_face_vertices;
    int* _gpu_object_types;
    float* _gpu_color_per_ray;
public:
    RayTracingCUDARenderer();
    void render(std::shared_ptr<Scene> scene,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<RayTracingOptions> options,
        pybind11::array_t<float, pybind11::array::c_style> buffer);
};
}
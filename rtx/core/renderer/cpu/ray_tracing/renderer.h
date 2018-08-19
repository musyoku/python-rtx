#pragma once
#include "../../../class/ray.h"
#include "../../../class/renderer.h"
#include "../../options/ray_tracing.h"
#include <array>
#include <memory>
#include <pybind11/numpy.h>
#include <random>

namespace rtx {
class RayTracingCPURenderer : public Renderer {
private:
    std::default_random_engine _normal_engine;
    std::normal_distribution<float> _normal_distribution;
    glm::vec3 compute_color(std::vector<std::shared_ptr<Mesh>>& mesh_array,
        std::unique_ptr<Ray>& ray,
        int current_reflection,
        int max_reflextions);

public:
    RayTracingCPURenderer();
    void render(std::shared_ptr<Scene> scene,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<RayTracingOptions> options,
        pybind11::array_t<float, pybind11::array::c_style> buffer);
    void render(std::shared_ptr<Scene> scene,
        std::shared_ptr<Camera> camera,
        std::shared_ptr<RayTracingOptions> options,
        unsigned char* buffer,
        int height,
        int width,
        int channels);
};
}
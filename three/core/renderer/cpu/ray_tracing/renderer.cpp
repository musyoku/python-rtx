#include "renderer.h"
#include "../../../class/ray.h"
#include "../../../geometry/sphere.h"
#include "../hit_test.h"
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace three {

using namespace cpu;
namespace py = pybind11;

void RayTracingCPURenderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingOptions> options,
    py::array_t<int, py::array::c_style> buffer)
{
    _height = buffer.shape(0);
    _width = buffer.shape(1);
    int channels = buffer.shape(2);
    if (channels != 3) {
        throw std::runtime_error("channels != 3");
    }
    auto pixel = buffer.mutable_unchecked<3>();
    std::vector<std::unique_ptr<Ray>> ray_array;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> supersampling_noise(0.0, 1.0);

    int ns = options->get_num_rays_per_pixel();

    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);
            glm::vec<3, int> color(0, 0, 0);

            for (int m = 0; m < ns; m++) {
                float ray_target_x = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
                float ray_target_y = 2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f;
                glm::vec3 direction = glm::vec3(ray_target_x, ray_target_y, -1.0f);

                std::unique_ptr<Ray> ray = std::make_unique<Ray>(origin, direction);
                bool hit_any = false;
                float min_distance = FLT_MAX;
                glm::vec3 reflection_normal;
                for (auto mesh : scene->_mesh_array) {
                    auto geometry = mesh->_geometry;

                    if (geometry->type() == GeometryTypeSphere) {
                        SphereGeometry* sphere = (SphereGeometry*)geometry.get();
                        float t = hit_sphere(mesh->_position, sphere->_radius, ray);
                        if (t <= 0.0f) {
                            continue;
                        }
                        if (min_distance <= t) {
                            continue;
                        }
                        min_distance = t;

                        reflection_normal = glm::normalize(ray->point(t) - mesh->_position);
                        hit_any = true;
                    }
                }

                if (hit_any) {
                    color.r += (reflection_normal.x + 1.0) * 127.5f;
                    color.g += (reflection_normal.y + 1.0) * 127.5f;
                    color.b += (reflection_normal.z + 1.0) * 127.5f;
                } else {
                    color.r += 255.0f;
                    color.g += 255.0f;
                    color.b += 255.0f;
                }
            }

            pixel(y, x, 0) = glm::clamp(int(color.r / float(ns)), 0, 255);
            pixel(y, x, 1) = glm::clamp(int(color.g / float(ns)), 0, 255);
            pixel(y, x, 2) = glm::clamp(int(color.b / float(ns)), 0, 255);
        }
    }
}
}
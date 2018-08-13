#include "renderer.h"
#include "../../../class/ray.h"
#include "../../../geometry/sphere.h"
#include "../hit_test.h"
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace three {
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

    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);
            glm::vec<3, int> color(0, 0, 0);

            int ns = options->get_num_rays_per_pixel();
            for (int m = 0; m < ns; m++) {
                float ray_target_x = 2.0f * float(x + supersampling_noise(generator)) / float(_width) - 1.0f;
                float ray_target_y = 2.0f * float(y + supersampling_noise(generator)) / float(_height) - 1.0f;
                glm::vec3 direction = glm::vec3(ray_target_x, ray_target_y, -1.0f);

                std::unique_ptr<Ray> ray = std::make_unique<Ray>(origin, direction);

                for (auto mesh : scene->_mesh_array) {
                    auto geometry = mesh->_geometry;

                    if (geometry->type() == GeometryTypeSphere) {
                        SphereGeometry* sphere = (SphereGeometry*)geometry.get();
                        float t = cpu::hit_sphere(mesh->_position, sphere->_radius, ray);
                        if (t > 0.0f) {
                            glm::vec3 normal = glm::normalize(ray->point(t) - mesh->_position);
                            color.r += (normal.x + 1.0) * 127.5f;
                            color.g += (normal.y + 1.0) * 127.5f;
                            color.b += (normal.z + 1.0) * 127.5f;
                        } else {
                            color.r += 255;
                            color.g += 255;
                            color.b += 255;
                        }
                    }
                }

                pixel(y, x, 0) = int(color.r / float(ns));
                pixel(y, x, 1) = int(color.y / float(ns));
                pixel(y, x, 2) = int(color.z / float(ns));
            }
        }
    }
}
}
#include "ray_tracing.h"
#include "../../class/ray.h"
#include "../../geometry/sphere.h"
#include "hit_test.h"
#include <iostream>
#include <memory>
#include <vector>

namespace three {
namespace py = pybind11;
void RayTracingCPURenderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    py::array_t<int, py::array::c_style> buffer)
{
    _height = buffer.shape(0);
    _width = buffer.shape(1);
    int channels = buffer.shape(2);
    if (channels != 3) {
        throw std::runtime_error("channels != 3");
    }
    auto pixel = buffer.mutable_unchecked<3>();
    glm::vec<3, int> color(0, 0, 0);
    std::vector<std::unique_ptr<Ray>> ray_array;

    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            glm::vec3 origin = glm::vec3(0.0f, 0.0f, 1.0f);

            float ray_target_x = 2.0f * float(x) / float(_width) - 1.0f;
            float ray_target_y = 2.0f * float(y) / float(_height) - 1.0f;
            glm::vec3 direction = glm::vec3(ray_target_x, ray_target_y, -1.0f);

            std::unique_ptr<Ray> ray = std::make_unique<Ray>(origin, direction);

            for (auto meth : scene->_mesh_array) {
                auto geometry = meth->_geometry;

                if (geometry->type() == GeometryTypeSphere) {
                    SphereGeometry* sphere = (SphereGeometry*)geometry.get();
                    if (cpu::hit_sphere(meth->_position, sphere->_radius, ray)) {
                        color.r = 255;
                        color.g = 0;
                        color.b = 0;
                    } else {
                        color.r = 255;
                        color.g = 255;
                        color.b = 255;
                    }
                }
            }

            pixel(y, x, 0) = color.r;
            pixel(y, x, 1) = color.g;
            pixel(y, x, 2) = color.b;
        }
    }
}
}
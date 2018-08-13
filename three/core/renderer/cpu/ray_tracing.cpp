#include "ray_tracing.h"
#include <iostream>

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

    for (int y = 0; y < _height; y++) {
        for (int x = 0; x < _width; x++) {
            for (int c = 0; c < channels; c++) {
                pixel(y, x, c) = (int)(float(x) / float(_width) * 255.0f);
            }
        }
    }
}
}
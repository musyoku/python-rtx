#include "renderer.h"
namespace three {
void Renderer::render(std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    pybind11::array_t<unsigned int, pybind11::array::c_style> buffer)
{
}
}
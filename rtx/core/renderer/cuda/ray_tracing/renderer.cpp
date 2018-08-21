#include "renderer.h"
#include "../../../geometry/sphere.h"
#include "../../../geometry/standard.h"
#include "cuda.h"
#include <memory>
#include <vector>

namespace rtx {

namespace py = pybind11;

RayTracingCUDARenderer::RayTracingCUDARenderer()
{
}
void RayTracingCUDARenderer::render(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<RayTracingOptions> options,
    py::array_t<float, py::array::c_style> buffer)
{
    a_cuda_kernerl_frontend();
}
}
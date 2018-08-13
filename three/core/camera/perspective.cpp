#include "perspective.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace three {
namespace py = pybind11;
PerspectiveCamera::PerspectiveCamera(py::tuple eye, py::tuple center, py::tuple up, float fov_rad, float aspect_ratio, float z_near, float z_far)
{
    this->look_at(eye, center, up);
    _fov_rad = fov_rad;
    _aspect_ratio = aspect_ratio;
    _z_near = z_near;
    _z_far = z_far;
}
}
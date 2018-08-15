#include "perspective.h"
#include <glm/glm.hpp>

namespace three {
namespace py = pybind11;
PerspectiveCamera::PerspectiveCamera(py::tuple eye, py::tuple center, py::tuple up, float fov_rad, float aspect_ratio, float z_near, float z_far)
{
    look_at(eye, center, up);
    _fov_rad = fov_rad;
    _aspect_ratio = aspect_ratio;
    _z_near = z_near;
    _z_far = z_far;
}

PerspectiveCamera::PerspectiveCamera(float (&eye)[3], float (&center)[3], float (&up)[3],
    float fov_rad, float aspect_ratio, float z_near, float z_far)
{
    look_at(eye, center, up);
    _fov_rad = fov_rad;
    _aspect_ratio = aspect_ratio;
    _z_near = z_near;
    _z_far = z_far;
}
}
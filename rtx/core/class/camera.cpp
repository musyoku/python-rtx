#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

namespace rtx {
namespace py = pybind11;
void Camera::look_at(py::tuple eye, py::tuple center, py::tuple up)
{
    _eye = glm::vec3(eye[0].cast<float>(), eye[1].cast<float>(), eye[2].cast<float>());
    _center = glm::vec3(center[0].cast<float>(), center[1].cast<float>(), center[2].cast<float>());
    _up = glm::vec3(up[0].cast<float>(), up[1].cast<float>(), up[2].cast<float>());
    _view_matrix = glm::lookAtRH(_eye, _center, _up);
    _updated = true;
}

void Camera::look_at(float (&eye)[3], float (&center)[3], float (&up)[3])
{
    _eye = glm::vec3(eye[0], eye[1], eye[2]);
    _center = glm::vec3(center[0], center[1], center[2]);
    _up = glm::vec3(up[0], up[1], up[2]);
    _view_matrix = glm::lookAtRH(_eye, _center, _up);
    _updated = true;
}
glm::mat4 Camera::view_matrix()
{
    return _view_matrix;
}
bool Camera::updated()
{
    return _updated;
}
void Camera::set_updated(bool updated)
{
    _updated = updated;
}
}
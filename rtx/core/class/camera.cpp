#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

namespace rtx {
namespace py = pybind11;
void Camera::look_at(py::tuple eye, py::tuple center, py::tuple up)
{
    _view_matrix = glm::lookAtRH(glm::vec3(eye[0].cast<float>(), eye[1].cast<float>(), eye[2].cast<float>()),
        glm::vec3(center[0].cast<float>(), center[1].cast<float>(), center[2].cast<float>()),
        glm::vec3(up[0].cast<float>(), up[1].cast<float>(), up[2].cast<float>()));
    _updated = true;
}

void Camera::look_at(float (&eye)[3], float (&center)[3], float (&up)[3])
{
    _view_matrix = glm::lookAtRH(glm::vec3(eye[0], eye[1], eye[2]),
        glm::vec3(center[0], center[1], center[2]),
        glm::vec3(up[0], up[1], up[2]));
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
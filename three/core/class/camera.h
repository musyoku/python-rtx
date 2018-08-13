#pragma once
#include <glm/glm.hpp>
#include <pybind11/pybind11.h>

namespace three {
class Camera {
public:
    glm::mat4 _view_matrix;
    void look_at(pybind11::tuple eye, pybind11::tuple center, pybind11::tuple up);
};
}
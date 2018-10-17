#pragma once
#include "../header/enum.h"
#include <glm/glm.hpp>
#include <pybind11/pybind11.h>

namespace rtx {
class Camera {
protected:
    bool _updated;

public:
    glm::mat4 _view_matrix;
    glm::vec3 _eye;
    glm::vec3 _center;
    glm::vec3 _up;

    glm::mat4 view_matrix();
    void look_at(pybind11::tuple eye, pybind11::tuple center, pybind11::tuple up);
    void look_at(float (&eye)[3], float (&center)[3], float (&up)[3]);
    bool updated();
    void set_updated(bool updated);
    virtual RTXCameraType type() const = 0;
};
}
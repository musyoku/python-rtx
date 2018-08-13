#pragma once
#include "geometry.h"
#include "material.h"
#include <glm/glm.hpp>
#include <memory>
#include <pybind11/pybind11.h>

namespace three {
class Mesh {
private:
    void update_model_matrix();

public:
    glm::vec3 _position; // xyz
    glm::vec3 _rotation_rad; // xyz
    glm::vec3 _scale; // xyz
    glm::mat4 _model_matrix;
    std::shared_ptr<Geometry> _geometry;
    std::shared_ptr<Material> _material;
    Mesh(std::shared_ptr<Geometry> geometry, std::shared_ptr<Material> material);
    void set_scale(pybind11::tuple scale);
    void set_position(pybind11::tuple position);
    void set_rotation(pybind11::tuple rotation_rad);
};
}
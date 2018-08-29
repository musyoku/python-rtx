#pragma once
#include "../header/glm.h"
#include "geometry.h"
#include "material.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace rtx {
class Mesh {
private:
    void update_model_matrix();
    glm::vec3f _position;
    glm::vec3f _rotation_rad;
    glm::vec3f _scale;
public:
    glm::mat4 _model_matrix;
    std::shared_ptr<Geometry> _geometry;
    std::shared_ptr<Material> _material;
    Mesh(std::shared_ptr<Geometry> geometry, std::shared_ptr<Material> material);
    void set_material(std::shared_ptr<Material> material);
    void set_scale(pybind11::tuple scale);
    void set_position(pybind11::tuple position);
    void set_rotation(pybind11::tuple rotation_rad);
    void set_position(float (&position)[3]);
};
}
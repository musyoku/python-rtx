#pragma once
#include "../../class/material.h"
#include <glm/glm.hpp>
#include <pybind11/pybind11.h>

namespace three {
class MeshLambertMaterial : public Material {
private:
    glm::vec3 _color;
    float _diffuse_reflectance;

public:
    // color: [0, 1]
    MeshLambertMaterial(pybind11::tuple color, float diffuse_reflectance);
    MeshLambertMaterial(float (&color)[3], float diffuse_reflectance);
    glm::vec3 reflect_color(glm::vec3& input_color) const override;
    glm::vec3 reflect_ray(glm::vec3& diffuse_vec, glm::vec3& specular_vec) const override;
    MaterialType type() const override;
};
}
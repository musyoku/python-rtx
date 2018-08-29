#pragma once
#include "../../class/material.h"
#include "../../header/glm.h"
#include <pybind11/pybind11.h>

namespace rtx {
class MeshLambertMaterial : public Material {
private:
    glm::vec3f _color;
    float _diffuse_reflectance;

public:
    // color: [0, 1]
    MeshLambertMaterial(pybind11::tuple color, float diffuse_reflectance);
    MeshLambertMaterial(float (&color)[3], float diffuse_reflectance);
    glm::vec3f reflect_color(glm::vec3f& input_color) const override;
    glm::vec3f reflect_ray(glm::vec3f& diffuse_vec, glm::vec3f& specular_vec) const override;
    glm::vec3f emit_color() const override;
    glm::vec3f color() const override;
    int type() const override;
};
}
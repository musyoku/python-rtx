#pragma once
#include "../../class/material.h"
#include "../../header/glm.h"
#include <pybind11/pybind11.h>

namespace rtx {
class MeshMetalMaterial : public Material {
private:
    float _roughness;
    float _specular_reflectance;

public:
    MeshMetalMaterial(float roughness, float specular_reflectance);
    glm::vec3f reflect_color(glm::vec3f& input_color) const override;
    glm::vec3f reflect_ray(glm::vec3f& diffuse_vec, glm::vec3f& specular_vec) const override;
    glm::vec3f emit_color() const override;
    glm::vec3f color() const override;
    int type() const override;
};
}
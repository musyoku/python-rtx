#pragma once
#include "../../class/material.h"
#include <glm/glm.hpp>
#include <pybind11/pybind11.h>

namespace rtx {
class MeshMetalMaterial : public Material {
private:
    float _roughness;
    float _specular_reflectance;

public:
    MeshMetalMaterial(float roughness, float specular_reflectance);
    glm::vec3 reflect_color(glm::vec3& input_color) const override;
    glm::vec3 reflect_ray(glm::vec3& diffuse_vec, glm::vec3& specular_vec) const override;
    glm::vec3 emit_color() const override;
    glm::vec3 color() const override;
    MaterialType type() const override;
};
}
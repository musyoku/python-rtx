#pragma once
#include <glm/glm.hpp>

enum MaterialType {
    MaterialTypeUnknown = 0,
    MaterialTypeLambert = 1,
    MaterialTypeMetal = 1,
};

namespace three {
class Material {
public:
    virtual glm::vec3 reflect_color(glm::vec3& input_color) const = 0;
    virtual glm::vec3 reflect_ray(glm::vec3& diffuse_vec, glm::vec3& specular_vec) const = 0;
    virtual MaterialType type() const = 0;
};
}
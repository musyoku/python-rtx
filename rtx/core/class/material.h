#pragma once
#include "../header/enum.h"
#include "../header/glm.h"
#include <glm/glm.hpp>

namespace rtx {
class Material {
public:
    virtual glm::vec3f color() const = 0;
    virtual glm::vec3f emit_color() const = 0;
    virtual glm::vec3f reflect_color(glm::vec3f& input_color) const = 0;
    virtual glm::vec3f reflect_ray(glm::vec3f& diffuse_vec, glm::vec3f& specular_vec) const = 0;
    virtual int type() const = 0;
};
}
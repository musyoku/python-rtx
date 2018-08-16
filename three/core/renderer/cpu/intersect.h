#pragma once
#include "../../class/ray.h"
#include <glm/glm.hpp>
#include <memory>

namespace three {
namespace cpu {
    float intersect_sphere(glm::vec3& center, float radius, std::unique_ptr<Ray>& ray);
    float intersect_triangle(glm::vec3& va, glm::vec3& vb, glm::vec3& vc, glm::vec3& face_normal, std::unique_ptr<Ray>& ray);
}
}
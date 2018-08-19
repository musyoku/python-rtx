#pragma once
#include "../../class/ray.h"
#include <glm/glm.hpp>
#include <memory>

namespace rtx {
namespace cpu {
    float intersect_sphere(const glm::vec3& center, const float radius, const std::unique_ptr<Ray>& ray);
    float intersect_triangle(const glm::vec3& va, const glm::vec3& vb, const glm::vec3& vc, const glm::vec3& face_normal, const std::unique_ptr<Ray>& ray);
}
}
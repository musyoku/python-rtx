#pragma once
#include "../../class/ray.h"
#include "../../header/glm.h"
#include <memory>

namespace rtx {
namespace cpu {
    float intersect_sphere(const glm::vec3f& center, const float radius, const std::unique_ptr<Ray>& ray);
    float intersect_triangle(const glm::vec3f& va, const glm::vec3f& vb, const glm::vec3f& vc, const std::unique_ptr<Ray>& ray);
}
}
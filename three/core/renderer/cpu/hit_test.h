#pragma once
#include "../../class/ray.h"
#include <glm/glm.hpp>
#include <memory>

namespace three {
namespace cpu {
    bool hit_sphere(glm::vec3& center, float radius, std::unique_ptr<Ray> &ray);
}
}
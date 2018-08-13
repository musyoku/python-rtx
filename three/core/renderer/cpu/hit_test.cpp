#include "hit_test.h"
#include "../../class/math.h"

namespace three {
namespace cpu {
    float hit_sphere(glm::vec3& position, float radius, std::unique_ptr<Ray>& ray)
    {
        glm::vec3 oc = ray->_origin - position;
        float a = glm::dot(ray->_direction, ray->_direction);
        float b = 2.0f * glm::dot(ray->_direction, oc);
        float c = glm::dot(oc, oc) - pow2(radius);
        float d = b * b - 4 * a * c;

        if (d < 0) {
            return -1.0f;
        } else {
            return (-b - sqrtf(d)) / (2.0f * a);
        }
    }
}
}
#include "intersect.h"
#include "../../class/math.h"
#include <iostream>

namespace rtx {
namespace cpu {
    float intersect_sphere(
        const glm::vec3& center,
        const float radius,
        const std::unique_ptr<Ray>& ray)
    {
        glm::vec3 oc = ray->_origin - center;
        const float a = glm::dot(ray->_direction, ray->_direction);
        const float b = 2.0f * glm::dot(ray->_direction, oc);
        const float c = glm::dot(oc, oc) - pow2(radius);
        const float d = b * b - 4.0f * a * c;

        if (d <= 0) {
            return -1.0f;
        }
        const float root = sqrtf(d);
        float t = (-b - root) / (2.0f * a);
        if (t > 0.001f) {
            return t;
        }
        t = (-b + root) / (2.0f * a);
        if (t > 0.001f) {
            return t;
        }
        return -1.0f;
    }
    // https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
    float intersect_triangle(
        const glm::vec3& va,
        const glm::vec3& vb,
        const glm::vec3& vc,
        const glm::vec3& face_normal,
        const std::unique_ptr<Ray>& ray)
    {

        const glm::vec3 edge_ba = vb - va;
        const glm::vec3 edge_ca = vc - va;

        const glm::vec3 q = glm::cross(ray->_direction, edge_ca);
        const float a = glm::dot(edge_ba, q);
        if (a < 1e-8f) {
            return -1.0f;
        }
        const glm::vec3 s = (ray->_origin - va) / a;
        const glm::vec3 r = glm::cross(s, edge_ba);
        const float b_x = glm::dot(s, q);
        const float b_y = glm::dot(r, ray->_direction);
        const float b_z = 1.0f - b_x - b_y;
        if (b_z < 0.0f || b_y < 0.0f || b_x < 0.0f) {
            return -1.0f;
        }

        return glm::dot(edge_ca, r);
    }
}
}
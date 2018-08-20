#include "intersect.h"
#include "../../class/math.h"
#include <xmmintrin.h>

namespace rtx {
namespace cpu {
    float intersect_sphere(
        const glm::vec3& center,
        const float radius,
        const std::unique_ptr<Ray>& ray)
    {
        const glm::vec3 oc = ray->_origin - center;
        const float a = glm::dot(ray->_direction, ray->_direction);
        const float b = 2.0f * glm::dot(ray->_direction, oc);
        const float c = glm::dot(oc, oc) - pow2(radius);
        const float d = b * b - 4.0f * a * c;

        if (d <= 0) {
            return -1.0f;
        }
        const float root = sqrtf(d);
        const float t0 = (-b - root) / (2.0f * a);
        if (t0 > 0.001f) {
            return t0;
        }
        const float t1 = (-b + root) / (2.0f * a);
        if (t1 > 0.001f) {
            return t1;
        }
        return -1.0f;
    }
    // https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
    float intersect_triangle(
        const glm::vec3& va,
        const glm::vec3& vb,
        const glm::vec3& vc,
        const std::unique_ptr<Ray>& ray)
    {
        const float eps = 0.0000001;
        const glm::vec3 edge_ba = vb - va;
        const glm::vec3 edge_ca = vc - va;

        const glm::vec3 h = glm::cross(ray->_direction, edge_ca);
        const float a = glm::dot(edge_ba, h);
        if (a > -eps && a < eps) {
            return -1.0f;
        }
        const float f = 1.0f / a;
        const glm::vec3 s = ray->_origin - va;
        const float u = f * glm::dot(s, h);
        if (u < 0.0f || u > 1.0f) {
            return -1.0f;
        }
        const glm::vec3 q = glm::cross(s, edge_ba);
        const float v = f * glm::dot(q, ray->_direction);
        if (v < 0.0f || u + v > 1.0f) {
            return -1.0f;
        }
        return f * glm::dot(edge_ca, q);
    }
}
}
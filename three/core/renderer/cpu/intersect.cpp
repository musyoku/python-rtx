#include "intersect.h"
#include "../../class/math.h"
#include <iostream>

namespace three {
namespace cpu {
    float intersect_sphere(glm::vec3& position, float radius, std::unique_ptr<Ray>& ray)
    {
        glm::vec3 oc = ray->_origin - position;
        float a = glm::dot(ray->_direction, ray->_direction);
        float b = 2.0f * glm::dot(ray->_direction, oc);
        float c = glm::dot(oc, oc) - pow2(radius);
        float d = b * b - 4.0f * a * c;

        if (d <= 0) {
            return -1.0f;
        }
        float root = sqrtf(d);
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
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
    float intersect_triangle(glm::vec3& va, glm::vec3& vb, glm::vec3& vc, glm::vec3& face_normal, std::unique_ptr<Ray>& ray)
    {

        glm::vec3 edge_ba = vb - va;
        glm::vec3 edge_ca = vc - va;

        auto q = glm::cross(ray->_direction, edge_ca);
        float a = glm::dot(edge_ba, q);
        auto s = (ray->_origin - va) / a;
        auto r = glm::cross(s, edge_ba);
        auto b = glm::vec3(glm::dot(s, q), glm::dot(r, ray->_direction), 0.0f);
        b.z = 1.0f - b.x - b.y;
        if (b.z < 0.0f || b.y < 0.0f || b.x < 0.0f) {
            return -1.0f;
        }

        float t = glm::dot(edge_ca, r);
        return t;


        // float t = (glm::dot(face_normal, va) - glm::dot(face_normal, ray->_origin)) / glm::dot(face_normal, ray->_direction);
        // glm::vec3 point = ray->point(t);
        // glm::vec3 edge0 = vb - va;
        // glm::vec3 edge1 = vc - vb;
        // glm::vec3 edge2 = va - vc;
        // glm::vec3 ca = point - va;
        // glm::vec3 cb = point - vb;
        // glm::vec3 cc = point - vc;

        // if (glm::dot(face_normal, glm::cross(edge0, ca)) > 0 && glm::dot(face_normal, glm::cross(edge1, cb)) > 0 && glm::dot(face_normal, glm::cross(edge2, cc)) > 0) {
        //     return t;
        // }
        // return -1.0f;
    }
}
}
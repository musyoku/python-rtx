#pragma once
#include "../class/geometry.h"
#include <glm/glm.hpp>

namespace rtx {
class SphereGeometry : public Geometry {
public:
    glm::vec4 _center;
    glm::vec4 _radius;
    SphereGeometry(float radius);
    int type() const override;
    int num_faces() const override;
    int num_vertices() const override;
    void pack_vertices(float*& buffer, int start, glm::mat4& transformation_matrix) const override;
};
}
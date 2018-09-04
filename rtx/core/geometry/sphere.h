#pragma once
#include "../class/geometry.h"
#include "../header/glm.h"

namespace rtx {
class SphereGeometry : public Geometry {
public:
    glm::vec4f _radius;
    glm::vec4f _center;
    SphereGeometry(float radius);
    int type() const override;
    int num_faces() const override;
    int num_vertices() const override;
    int serialize_vertices(rtx::array<float>& buffer, int start) const override;
    int serialize_faces(rtx::array<int>& buffer, int start, int vertex_index_offset) const override;
    std::shared_ptr<Geometry> transoform(glm::mat4& transformation_matrix) const override;
};
}
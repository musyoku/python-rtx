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
    void serialize_vertices(rtx::array<RTXVertex>& array, int offset) const override;
    void serialize_faces(rtx::array<RTXFace>& array, int array_offset, int vertex_index_offset) const override;
    std::shared_ptr<Object> transoform(glm::mat4& transformation_matrix) const override;
};
}
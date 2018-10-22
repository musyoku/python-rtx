#pragma once
#include "../class/geometry.h"
#include "../header/array.h"
#include "../header/glm.h"

namespace rtx {
class SphereGeometry : public Geometry {
protected:
    float _radius;

public:
    SphereGeometry(float radius);
    float radius();
    glm::vec4f center();
    int type() const override;
    int num_faces() const override;
    int num_vertices() const override;
    void serialize_vertices(rtx::array<rtxVertex>& array, int offset) const override;
    void serialize_faces(rtx::array<rtxFaceVertexIndex>& array, int array_offset) const override;
    std::shared_ptr<Geometry> transoform(glm::mat4& transformation_matrix) const override;
};
}
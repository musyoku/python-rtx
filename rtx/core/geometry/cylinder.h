#pragma once
#include "../class/geometry.h"
#include "../header/array.h"
#include "../header/glm.h"

namespace rtx {
class CylinderGeometry : public Geometry {
protected:
    float _radius;
    float _y_min;
    float _y_max;
    float _height;
    glm::mat4f _transformation_matrix;

public:
    CylinderGeometry(float radius, float height);
    int type() const override;
    int num_faces() const override;
    int num_vertices() const override;
    void serialize_vertices(rtx::array<rtxVertex>& array, int offset) const override;
    void serialize_faces(rtx::array<rtxFaceVertexIndex>& array, int offset) const override;
    void set_transformation_matrix(glm::mat4f& transformation_matrix);
    std::shared_ptr<Geometry> transoform(glm::mat4& transformation_matrix) const override;
};
}
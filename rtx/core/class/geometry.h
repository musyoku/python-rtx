#pragma once
#include "../header/array.h"
#include "../header/enum.h"
#include "../header/glm.h"
#include <vector>

namespace rtx {
class Geometry {
public:
    int _bvh_num_max_triangles_per_node = 0;
    glm::vec4f _aabb_min;
    glm::vec4f _center;
    glm::vec4f _aabb_max;
    int num_bvh_split();
    virtual int type() const = 0;
    virtual int num_faces() const = 0;
    virtual int num_vertices() const = 0;
    virtual int serialize_vertices(rtx::array<float>& buffer, int start) const = 0;
    virtual int serialize_faces(rtx::array<int>& buffer, int start, int vertex_index_offset) const = 0;
    virtual void compute_axis_aligned_bounding_box() = 0;
    virtual std::shared_ptr<Geometry> transoform(glm::mat4& transformation_matrix) const = 0;
};
}
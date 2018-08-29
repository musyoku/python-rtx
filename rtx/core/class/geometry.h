#pragma once
#include "../bvh/geometry.h"
#include "../header/array.h"
#include "../header/enum.h"
#include <glm/glm.hpp>
#include <vector>

namespace rtx {
class Geometry {
protected:
    int _num_bvh_split = 1;
    std::shared_ptr<bvh::geometry::GeometryBVH> _bvh;

public:
    std::shared_ptr<bvh::geometry::GeometryBVH> bvh();
    int num_bvh_split();
    virtual int type() const = 0;
    virtual int num_faces() const = 0;
    virtual int num_vertices() const = 0;
    virtual int serialize_vertices(rtx::array<float>& buffer, int start, glm::mat4& transformation_matrix) const = 0;
    virtual int serialize_faces(rtx::array<int>& buffer, int start, int vertex_index_offset) const = 0;
};
}
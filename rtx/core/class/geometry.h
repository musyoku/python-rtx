#pragma once
#include "array.h"
#include "enum.h"
#include <glm/glm.hpp>

namespace rtx {
class Geometry {
private:
    int _num_bvh_split = 1;

public:
    virtual int type() const = 0;
    virtual int num_bvh_split() const = 0;
    virtual int num_faces() const = 0;
    virtual int num_vertices() const = 0;
    virtual int serialize_vertices(rtx::array<float>& buffer, int start, glm::mat4& transformation_matrix) const = 0;
    virtual int serialize_faces(rtx::array<int>& buffer, int start, int vertex_index_offset) const = 0;
};
}
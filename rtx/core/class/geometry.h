#pragma once
#include "array.h"
#include "enum.h"
#include <glm/glm.hpp>

namespace rtx {
class Geometry {
public:
    virtual int type() const = 0;
    virtual int num_faces() const = 0;
    virtual int num_vertices() const = 0;
    virtual int pack_vertices(rtx::array<float>& buffer, int start, glm::mat4& transformation_matrix) const = 0;
};
}
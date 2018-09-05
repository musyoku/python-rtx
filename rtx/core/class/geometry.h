#pragma once
#include "../header/array.h"
#include "../header/enum.h"
#include "../header/glm.h"
#include "../header/struct.h"
#include <memory>

namespace rtx {
class Geometry {
public:
    virtual int bvh_max_triangles_per_node() const;
    virtual bool bvh_enabled() const;
    virtual int type() const = 0;
    virtual int num_faces() const = 0;
    virtual int num_vertices() const = 0;
    virtual void serialize_vertices(rtx::array<RTXGeometryVertex>& array, int offset) const = 0;
    virtual void serialize_faces(rtx::array<RTXGeometryFace>& array, int array_offset, int vertex_index_offset) const = 0;
    virtual std::shared_ptr<Geometry> transoform(glm::mat4& transformation_matrix) const = 0;
};
}
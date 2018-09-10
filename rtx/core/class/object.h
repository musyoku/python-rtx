#pragma once
#include "../header/array.h"
#include "../header/struct.h"
#include "../header/glm.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace rtx {
class Object {
public:
    void set_position(pybind11::tuple position);
    void set_position(float (&position)[3]);
    void set_rotation(pybind11::tuple rotation_rad);
    virtual int bvh_max_triangles_per_node() const;
    virtual bool bvh_enabled() const;
    virtual bool is_light() const = 0;
    virtual int type() const = 0;
    virtual int num_faces() const = 0;
    virtual int num_vertices() const = 0;
    virtual void serialize_vertices(rtx::array<RTXVertex>& array, int offset) const = 0;
    virtual void serialize_faces(rtx::array<RTXFace>& array, int array_offset, int vertex_index_offset) const = 0;
    virtual std::shared_ptr<Object> transoform(glm::mat4& transformation_matrix) const = 0;
};
}
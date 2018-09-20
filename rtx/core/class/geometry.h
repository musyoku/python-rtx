#pragma once
#include "../header/array.h"
#include "../header/glm.h"
#include "../header/struct.h"
#include <pybind11/pybind11.h>

namespace rtx {
class Geometry {
protected:
    void update_model_matrix();
    glm::vec3f _position;
    glm::vec3f _rotation_rad;
    glm::vec3f _scale;
    glm::mat4 _model_matrix;
    bool _updated;

public:
    Geometry();
    void set_scale(pybind11::tuple scale);
    void set_scale(float (&scale)[3]);
    void set_position(pybind11::tuple position);
    void set_position(float (&position)[3]);
    void set_rotation(pybind11::tuple rotation_rad);
    void set_rotation(float (&rotation)[3]);
    bool updated();
    void set_updated(bool updated);
    glm::mat4f model_matrix();
    virtual int bvh_max_triangles_per_node() const;
    virtual int type() const = 0;
    virtual int num_faces() const = 0;
    virtual int num_vertices() const = 0;
    virtual void serialize_vertices(rtx::array<rtxVertex>& array, int offset) const = 0;
    virtual void serialize_faces(rtx::array<rtxFaceVertexIndex>& array, int array_offset) const = 0;
    virtual std::shared_ptr<Geometry> transoform(glm::mat4& transformation_matrix) const = 0;
};
}
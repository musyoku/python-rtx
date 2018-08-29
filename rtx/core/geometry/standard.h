#pragma once
#include "../class/geometry.h"
#include "../header/glm.h"
#include <memory>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>

namespace rtx {
class StandardGeometry : public Geometry {
private:
    void init(pybind11::array_t<int, pybind11::array::c_style> face_vertex_indeces,
        pybind11::array_t<float, pybind11::array::c_style> vertices,
        int num_bvh_split);
    int _num_bvh_split;

public:
    // 各面を構成する3頂点のインデックス
    // Index of three vertices constituting each face.
    std::vector<glm::vec3i> _face_vertex_indices_array;
    // 頂点の配列
    // 頂点は同次座標系で表される
    // Vertices are 4-dimensional points in projective coordinates system.
    std::vector<glm::vec4f> _vertex_array;
    StandardGeometry();
    StandardGeometry(pybind11::array_t<int, pybind11::array::c_style> face_vertex_indeces,
        pybind11::array_t<float, pybind11::array::c_style> vertices);
    StandardGeometry(pybind11::array_t<int, pybind11::array::c_style> face_vertex_indeces,
        pybind11::array_t<float, pybind11::array::c_style> vertices,
        int num_bvh_split);
    int type() const override;
    int num_faces() const override;
    int num_vertices() const override;
    int serialize_vertices(rtx::array<float>& buffer, int start, glm::mat4& transformation_matrix) const override;
    int serialize_faces(rtx::array<int>& buffer, int start, int vertex_index_offset) const override;
};
}
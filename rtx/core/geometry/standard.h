#pragma once
#include "../class/geometry.h"
#include <glm/glm.hpp>
#include <memory>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>

namespace rtx {
class StandardGeometry : public Geometry {
public:
    // 各面を構成する3頂点のインデックス
    // Index of three vertices constituting each face.
    std::vector<glm::vec<3, int>> _face_vertex_indices_array;
    // 頂点の配列
    // 頂点は同次座標系で表される
    // Vertices are 4-dimensional points in projective coordinates system.
    std::vector<glm::vec4> _vertex_array;
    // BVHの各分割のインデックス
    // Index of each division of BVH.
    std::vector<int> _bvh_indices;
    // 各分割に含まれる頂点の開始インデックスと終了インデックス
    // Start index and end index of the vertex included in each division.
    std::vector<std::pair<int, int>> _bvh_start_end_vertex_indices;
    StandardGeometry();
    StandardGeometry(pybind11::array_t<int, pybind11::array::c_style> face_vertex_indeces, pybind11::array_t<float, pybind11::array::c_style> vertices);
    int type() const override;
    int num_faces() const override;
    int num_vertices() const override;
    int serialize_vertices(rtx::array<float>& buffer, int start, glm::mat4& transformation_matrix) const override;
    int serialize_faces(rtx::array<int>& buffer, int start, int offset) const override;
};
}
#pragma once
#include "../class/geometry.h"
#include "../header/array.h"
#include "../header/enum.h"
#include "../header/glm.h"
#include <memory>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>

namespace rtx {
class StandardGeometry : public Geometry {
protected:
    int _bvh_max_triangles_per_node = BVH_DEFAULT_TRIANGLES_PER_NODE;
    void init(pybind11::array_t<int, pybind11::array::c_style> face_vertex_indeces,
        pybind11::array_t<float, pybind11::array::c_style> vertices,
        int bvh_max_triangles_per_node);

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
        int bvh_max_triangles_per_node);
    int type() const override;
    int num_faces() const override;
    int num_vertices() const override;
    void add_face(glm::vec3i face);
    void add_vertex(glm::vec3f vertex);
    void set_bvh_max_triangles_per_node(int bvh_max_triangles_per_node);
    void serialize_vertices(rtx::array<rtxVertex>& array, int offset) const override;
    void serialize_faces(rtx::array<rtxFaceVertexIndex>& array, int array_offset) const override;
    std::shared_ptr<Geometry> transoform(glm::mat4& transformation_matrix) const override;
    int bvh_max_triangles_per_node() const override;
};
}
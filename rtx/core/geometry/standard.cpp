#include "standard.h"
#include "../header/enum.h"
#include "../header/glm.h"
#include <cassert>
#include <cfloat>
#include <iostream>
#include <omp.h>

namespace rtx {
namespace py = pybind11;
StandardGeometry::StandardGeometry()
    : Geometry()
{
}
StandardGeometry::StandardGeometry(
    py::array_t<int, py::array::c_style> face_vertex_indeces,
    py::array_t<float, py::array::c_style> vertices)
    : Geometry()
{
    init(face_vertex_indeces, vertices, BVH_DEFAULT_TRIANGLES_PER_NODE);
}
StandardGeometry::StandardGeometry(
    py::array_t<int, py::array::c_style> face_vertex_indeces,
    py::array_t<float, py::array::c_style> vertices,
    int bvh_max_triangles_per_node)
    : Geometry()
{
    init(face_vertex_indeces, vertices, bvh_max_triangles_per_node);
}
void StandardGeometry::init(
    py::array_t<int, py::array::c_style> np_face_vertex_indeces,
    py::array_t<float, py::array::c_style> np_vertices,
    int bvh_max_triangles_per_node)
{
    if (np_face_vertex_indeces.ndim() != 2) {
        throw std::runtime_error("(num_np_face_vertex_indeces.ndim() != 2) -> false");
    }
    if (np_vertices.ndim() != 2) {
        throw std::runtime_error("(num_np_vertices.ndim() != 2) -> false");
    }
    _bvh_max_triangles_per_node = bvh_max_triangles_per_node;
    int num_faces = np_face_vertex_indeces.shape(0);
    int num_vertices = np_vertices.shape(0);
    int ndim_vertex = np_vertices.shape(1);
    if (ndim_vertex != 3 && ndim_vertex != 4) {
        throw std::runtime_error("(ndim_vertex != 3 && ndim_vertex != 4) -> false");
    }
    auto faces = np_face_vertex_indeces.mutable_unchecked<2>();
    auto vertices = np_vertices.mutable_unchecked<2>();
    for (int n = 0; n < num_faces; n++) {
        _face_vertex_indices_array.emplace_back(glm::vec3i(faces(n, 0), faces(n, 1), faces(n, 2)));
    }
    for (int n = 0; n < num_vertices; n++) {
        if (ndim_vertex == 3) {
            _vertex_array.emplace_back(glm::vec4f(vertices(n, 0), vertices(n, 1), vertices(n, 2), 1.0f));
        } else {
            _vertex_array.emplace_back(glm::vec4f(vertices(n, 0), vertices(n, 1), vertices(n, 2), vertices(n, 3)));
        }
    }
}
void StandardGeometry::add_face(glm::vec3i face)
{
    _face_vertex_indices_array.emplace_back(face);
}
void StandardGeometry::add_vertex(glm::vec3f vertex)
{
    _vertex_array.emplace_back(glm::vec4f(vertex.x, vertex.y, vertex.z, 1.0f));
}
void StandardGeometry::set_bvh_max_triangles_per_node(int bvh_max_triangles_per_node)
{
    _bvh_max_triangles_per_node = bvh_max_triangles_per_node;
}
int StandardGeometry::type() const
{
    return RTXGeometryTypeStandard;
}
int StandardGeometry::num_faces() const
{
    return _face_vertex_indices_array.size();
}
int StandardGeometry::num_vertices() const
{
    return _vertex_array.size();
}
void StandardGeometry::serialize_vertices(rtx::array<rtxVertex>& buffer, int array_offset) const
{
    for (unsigned int j = 0; j < _vertex_array.size(); j++) {
        auto& vertex = _vertex_array[j];
        buffer[j + array_offset] = { vertex.x, vertex.y, vertex.z, vertex.w };
    }
}
void StandardGeometry::serialize_faces(rtx::array<rtxFaceVertexIndex>& buffer, int array_offset) const
{
    for (unsigned int j = 0; j < _face_vertex_indices_array.size(); j++) {
        auto& face = _face_vertex_indices_array[j];
        buffer[j + array_offset] = { face[0], face[1], face[2], -1 };
    }
}
std::shared_ptr<Geometry> StandardGeometry::transoform(glm::mat4& transformation_matrix) const
{
    auto geometry = std::make_shared<StandardGeometry>();
    geometry->_bvh_max_triangles_per_node = _bvh_max_triangles_per_node;
    geometry->_face_vertex_indices_array = _face_vertex_indices_array;
    geometry->_vertex_array.resize(_vertex_array.size());

#pragma omp parallel for
    for (unsigned int index = 0; index < _vertex_array.size(); index++) {
        auto& vertex = _vertex_array[index];
        glm::vec4f v = transformation_matrix * vertex;
        geometry->_vertex_array[index] = v;
    }
    return geometry;
}
int StandardGeometry::bvh_max_triangles_per_node() const
{
    return _bvh_max_triangles_per_node;
}
}
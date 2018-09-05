#include "standard.h"
#include <cassert>
#include <cfloat>
#include <iostream>

namespace rtx {
namespace py = pybind11;
StandardGeometry::StandardGeometry()
{
}
StandardGeometry::StandardGeometry(
    py::array_t<int, py::array::c_style> face_vertex_indeces,
    py::array_t<float, py::array::c_style> vertices)
{
    init(face_vertex_indeces, vertices, -1);
}
StandardGeometry::StandardGeometry(
    py::array_t<int, py::array::c_style> face_vertex_indeces,
    py::array_t<float, py::array::c_style> vertices,
    int bvh_max_triangles_per_node)
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
        glm::vec3i face = glm::vec3i(faces(n, 0), faces(n, 1), faces(n, 2));
        _face_vertex_indices_array.emplace_back(face);
    }
    for (int n = 0; n < num_vertices; n++) {
        if (ndim_vertex == 3) {
            glm::vec4f vertex = glm::vec4f(vertices(n, 0), vertices(n, 1), vertices(n, 2), 1.0f);
            _vertex_array.emplace_back(vertex);
        } else {
            glm::vec4f vertex = glm::vec4f(vertices(n, 0), vertices(n, 1), vertices(n, 2), vertices(n, 3));
            _vertex_array.emplace_back(vertex);
        }
    }
}
int StandardGeometry::type() const
{
    return RTX_GEOMETRY_TYPE_STANDARD;
}
int StandardGeometry::num_faces() const
{
    return _face_vertex_indices_array.size();
}
int StandardGeometry::num_vertices() const
{
    return _vertex_array.size();
}
void StandardGeometry::serialize_vertices(rtx::array<RTXGeometryVertex>& buffer, int array_offset) const
{
    for (int j = 0; j < _vertex_array.size(); j++) {
        auto& vertex = _vertex_array[j];
        buffer[j + array_offset] = { vertex.x, vertex.y, vertex.z };
    }
}

void StandardGeometry::serialize_faces(rtx::array<RTXGeometryFace>& buffer, int array_offset, int vertex_index_offset) const
{
    for (int j = 0; j < _face_vertex_indices_array.size(); j++) {
        auto& face = _face_vertex_indices_array[j];
        buffer[j + array_offset] = { face[0] + vertex_index_offset, face[1] + vertex_index_offset, face[2] + vertex_index_offset };
    }
}
std::shared_ptr<Geometry> StandardGeometry::transoform(glm::mat4& transformation_matrix) const
{
    auto geometry = std::make_shared<StandardGeometry>();
    geometry->_bvh_max_triangles_per_node = _bvh_max_triangles_per_node;
    geometry->_face_vertex_indices_array = _face_vertex_indices_array;
    for (auto vertex : _vertex_array) {
        glm::vec4f v = transformation_matrix * vertex;
        geometry->_vertex_array.emplace_back(v);
    }
    return geometry;
}
bool StandardGeometry::bvh_enabled() const
{
    return _bvh_max_triangles_per_node > 0;
}
int StandardGeometry::bvh_max_triangles_per_node() const
{
    return _bvh_max_triangles_per_node;
}
}
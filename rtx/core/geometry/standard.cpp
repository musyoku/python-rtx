#include "standard.h"

namespace rtx {
namespace py = pybind11;
StandardGeometry::StandardGeometry()
{
}
StandardGeometry::StandardGeometry(pybind11::array_t<int, pybind11::array::c_style> np_face_vertex_indeces, pybind11::array_t<float, pybind11::array::c_style> np_vertices)
{
    if (np_face_vertex_indeces.ndim() != 2) {
        throw std::runtime_error("num_np_face_vertex_indeces.ndim() != 2");
    }
    if (np_vertices.ndim() != 2) {
        throw std::runtime_error("num_np_vertices.ndim() != 2");
    }
    int num_faces = np_face_vertex_indeces.shape(0);
    int num_vertices = np_vertices.shape(0);
    int ndim_vertex = np_vertices.shape(1);
    if (ndim_vertex != 4) {
        throw std::runtime_error("ndim_vertex != 4");
    }
    auto faces = np_face_vertex_indeces.mutable_unchecked<2>();
    auto vertices = np_vertices.mutable_unchecked<2>();
    for (int n = 0; n < num_faces; n++) {
        glm::vec<3, int> face = glm::vec<3, int>(faces(n, 0), faces(n, 1), faces(n, 2));
        _face_vertex_indices_array.emplace_back(face);
    }
    for (int n = 0; n < num_vertices; n++) {
        glm::vec4 vertex = glm::vec4(vertices(n, 0), vertices(n, 1), vertices(n, 2), vertices(n, 3));
        _vertex_array.emplace_back(vertex);
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
int StandardGeometry::serialize_vertices(rtx::array<float>& buffer, int start, glm::mat4& transformation_matrix) const
{
    int pos = start;
    for (auto& vertex : _vertex_array) {
        glm::vec4 v = transformation_matrix * vertex;
        buffer[pos + 0] = v.x;
        buffer[pos + 1] = v.y;
        buffer[pos + 2] = v.z;
        buffer[pos + 3] = v.w;
        pos += 4;
    }
    return pos;
}

int StandardGeometry::serialize_faces(rtx::array<int>& buffer, int start, int offset) const
{
    int pos = start;
    for (auto& face : _face_vertex_indices_array) {
        buffer[pos + 0] = face[0] + offset;
        buffer[pos + 1] = face[1] + offset;
        buffer[pos + 2] = face[2] + offset;
        buffer[pos + 3] = -1;
        pos += 4;
    }
    return pos;
}
}
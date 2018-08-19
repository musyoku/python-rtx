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
    if (ndim_vertex != 3) {
        throw std::runtime_error("ndim_vertex != 3");
    }
    auto faces = np_face_vertex_indeces.mutable_unchecked<2>();
    auto vertices = np_vertices.mutable_unchecked<2>();
    for (int n = 0; n < num_faces; n++) {
        glm::vec<3, int> face = glm::vec<3, int>(faces(n, 0), faces(n, 1), faces(n, 2));
        _face_vertex_indices_array.emplace_back(face);
    }
    for (int n = 0; n < num_vertices; n++) {
        glm::vec3 vertex = glm::vec3(vertices(n, 0), vertices(n, 1), vertices(n, 2));
        _vertex_array.emplace_back(vertex);
    }
    for (auto& face : _face_vertex_indices_array) {
        glm::vec3& va = _vertex_array[face[0]];
        glm::vec3& vb = _vertex_array[face[1]];
        glm::vec3& vc = _vertex_array[face[2]];
        glm::vec3 vba = vb - va;
        glm::vec3 vca = vc - va;
        glm::vec3 normal = glm::normalize(glm::cross(vba, vca));
        _face_normal_array.emplace_back(normal);
    }
}
GeometryType StandardGeometry::type()
{
    return GeometryTypeStandard;
}
}
#include "plain.h"

namespace rtx {
namespace py = pybind11;
PlainGeometry::PlainGeometry(float width, float height)
{
    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(0, 1, 2));
    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(2, 1, 3));

    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, -height / 2.0f, 0.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, -height / 2.0f, 0.0f));
    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, height / 2.0f, 0.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, height / 2.0f, 0.0f));

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
GeometryType PlainGeometry::type()
{
    return GeometryTypeStandard;
}
}
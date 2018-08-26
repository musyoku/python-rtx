#include "box.h"

namespace rtx {
// width:  width of the sides on the X axis
// height: height of the sides on the Y axis
// depth:  depth of the sides on the Z axis
BoxGeometry::BoxGeometry(float width, float height, float depth)
{
    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(0, 1, 2));
    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(2, 1, 3));

    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(4, 5, 6));
    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(6, 5, 7));

    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(8, 9, 10));
    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(10, 9, 11));

    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(12, 13, 14));
    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(14, 13, 15));

    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(16, 17, 18));
    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(18, 17, 19));
    
    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(20, 21, 22));
    _face_vertex_indices_array.emplace_back(glm::vec<3, int>(22, 21, 23));

    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, -height / 2.0f, depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, -height / 2.0f, depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, height / 2.0f, depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f));

    _vertex_array.emplace_back(glm::vec3(width / 2.0f, -height / 2.0f, depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f));

    _vertex_array.emplace_back(glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f));

    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, -height / 2.0f, depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, height / 2.0f, depth / 2.0f));

    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, height / 2.0f, depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f));

    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(-width / 2.0f, -height / 2.0f, depth / 2.0f));
    _vertex_array.emplace_back(glm::vec3(width / 2.0f, -height / 2.0f, depth / 2.0f));

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
GeometryType BoxGeometry::type()
{
    return GeometryTypeStandard;
}
}
#include "box.h"

namespace rtx {
// width:  width of the sides on the X axis
// height: height of the sides on the Y axis
// depth:  depth of the sides on the Z axis
BoxGeometry::BoxGeometry(float width, float height, float depth)
{
    _face_vertex_indices_array.emplace_back(glm::vec3i(0, 1, 2));
    _face_vertex_indices_array.emplace_back(glm::vec3i(2, 1, 3));

    _face_vertex_indices_array.emplace_back(glm::vec3i(4, 5, 6));
    _face_vertex_indices_array.emplace_back(glm::vec3i(6, 5, 7));

    _face_vertex_indices_array.emplace_back(glm::vec3i(8, 9, 10));
    _face_vertex_indices_array.emplace_back(glm::vec3i(10, 9, 11));

    _face_vertex_indices_array.emplace_back(glm::vec3i(12, 13, 14));
    _face_vertex_indices_array.emplace_back(glm::vec3i(14, 13, 15));

    _face_vertex_indices_array.emplace_back(glm::vec3i(16, 17, 18));
    _face_vertex_indices_array.emplace_back(glm::vec3i(18, 17, 19));

    _face_vertex_indices_array.emplace_back(glm::vec3i(20, 21, 22));
    _face_vertex_indices_array.emplace_back(glm::vec3i(22, 21, 23));

    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, -height / 2.0f, depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, -height / 2.0f, depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, height / 2.0f, depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, height / 2.0f, depth / 2.0f, 1.0f));

    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, -height / 2.0f, depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, -height / 2.0f, -depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, height / 2.0f, depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, height / 2.0f, -depth / 2.0f, 1.0f));

    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, -height / 2.0f, -depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, -height / 2.0f, -depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, height / 2.0f, -depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, height / 2.0f, -depth / 2.0f, 1.0f));

    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, -height / 2.0f, -depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, -height / 2.0f, depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, height / 2.0f, -depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, height / 2.0f, depth / 2.0f, 1.0f));

    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, height / 2.0f, depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, height / 2.0f, depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, height / 2.0f, -depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, height / 2.0f, -depth / 2.0f, 1.0f));

    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, -height / 2.0f, -depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, -height / 2.0f, -depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, -height / 2.0f, depth / 2.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, -height / 2.0f, depth / 2.0f, 1.0f));
}
}
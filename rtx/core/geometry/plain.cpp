#include "plain.h"

namespace rtx {
PlainGeometry::PlainGeometry(float width, float height)
{
    _face_vertex_indices_array.emplace_back(glm::vec3i(0, 1, 2));
    _face_vertex_indices_array.emplace_back(glm::vec3i(2, 1, 3));

    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, -height / 2.0f, 0.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, -height / 2.0f, 0.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, height / 2.0f, 0.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, height / 2.0f, 0.0f, 1.0f));

    _bvh = std::make_unique<bvh::geometry::GeometryBVH>(_face_vertex_indices_array, _vertex_array, 1);
}
}
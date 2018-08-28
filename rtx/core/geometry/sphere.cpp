#include "sphere.h"

namespace rtx {
SphereGeometry::SphereGeometry(float radius)
{
    _center = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    _radius = glm::vec4(radius, radius, radius, 1.0f);
}
int SphereGeometry::type() const
{
    return RTX_GEOMETRY_TYPE_SPHERE;
}
int SphereGeometry::num_faces() const
{
    return 1;
}
int SphereGeometry::num_vertices() const
{
    // center + radius
    return 2;
}
int SphereGeometry::serialize_vertices(rtx::array<float>& buffer, int start, glm::mat4& transformation_matrix) const
{
    int pos = start;
    // face_id = 0
    glm::vec4 center = transformation_matrix * _center;
    buffer[pos + 0] = center.x;
    buffer[pos + 1] = center.y;
    buffer[pos + 2] = center.z;
    buffer[pos + 3] = center.w;
    pos += 4;

    // face_id = 1
    glm::vec4 radius = transformation_matrix * _radius;
    buffer[pos + 0] = radius.x;
    buffer[pos + 1] = radius.y;
    buffer[pos + 2] = radius.z;
    buffer[pos + 3] = radius.w;
    pos += 4;

    return pos;
}

int SphereGeometry::serialize_faces(rtx::array<int>& buffer, int start, int vertex_index_offset) const
{
    int pos = start;
    buffer[pos + 0] = 0 + vertex_index_offset;
    buffer[pos + 1] = 1 + vertex_index_offset;
    buffer[pos + 2] = -1;
    buffer[pos + 3] = -1;
    pos += 4;
    return pos;
}
}
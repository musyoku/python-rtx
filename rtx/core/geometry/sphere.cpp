#include "sphere.h"

namespace rtx {
SphereGeometry::SphereGeometry(float radius)
{
    _center = glm::vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    _radius = glm::vec4f(radius, radius, radius, 0.0f);
    compute_axis_aligned_bounding_box();
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
int SphereGeometry::serialize_vertices(rtx::array<float>& buffer, int start) const
{
    int pos = start;
    // face_id = 0
    buffer[pos + 0] = _center.x;
    buffer[pos + 1] = _center.y;
    buffer[pos + 2] = _center.z;
    buffer[pos + 3] = _center.w;
    pos += 4;

    // face_id = 1
    buffer[pos + 0] = _radius.x;
    buffer[pos + 1] = _radius.y;
    buffer[pos + 2] = _radius.z;
    buffer[pos + 3] = _radius.w;
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
std::shared_ptr<Geometry> SphereGeometry::transoform(glm::mat4& transformation_matrix) const
{
    auto sphere = std::make_shared<SphereGeometry>(_radius[0]);
    sphere->_center = transformation_matrix * _center;
    sphere->compute_axis_aligned_bounding_box();
    return sphere;
}
void SphereGeometry::compute_axis_aligned_bounding_box()
{
    _aabb_min = _center - _radius;
    _aabb_max = _center + _radius;
}
}
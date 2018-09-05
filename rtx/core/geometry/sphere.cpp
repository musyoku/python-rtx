#include "sphere.h"

namespace rtx {
SphereGeometry::SphereGeometry(float radius)
{
    _center = glm::vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    _radius = glm::vec4f(radius, radius, radius, 0.0f);
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
void SphereGeometry::serialize_vertices(rtx::array<RTXGeometryVertex>& array, int offset) const
{
    array[0 + offset] = { _center.x, _center.y, _center.z };
    array[1 + offset] = { _radius.x, _radius.y, _radius.z };
}
void SphereGeometry::serialize_faces(rtx::array<RTXGeometryFace>& array, int array_offset, int vertex_index_offset) const
{
    array[0 + array_offset] = { 0 + vertex_index_offset, 1 + vertex_index_offset, -1 };
}
std::shared_ptr<Geometry> SphereGeometry::transoform(glm::mat4& transformation_matrix) const
{
    auto sphere = std::make_shared<SphereGeometry>(_radius[0]);
    sphere->_center = transformation_matrix * _center;
    return sphere;
}
}
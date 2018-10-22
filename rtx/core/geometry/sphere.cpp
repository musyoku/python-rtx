#include "sphere.h"
#include "../header/enum.h"

namespace rtx {
SphereGeometry::SphereGeometry(float radius)
    : Geometry()
{
    _radius = radius;
}
float SphereGeometry::radius()
{
    return _radius;
}
int SphereGeometry::type() const
{
    return RTXGeometryTypeSphere;
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
void SphereGeometry::serialize_vertices(rtx::array<rtxVertex>& array, int offset) const
{
    array[0 + offset] = { _position.x, _position.y, _position.z, 1.0f };
    array[1 + offset] = { _radius, -1.0f, -1.0f, -1.0f };
}
void SphereGeometry::serialize_faces(rtx::array<rtxFaceVertexIndex>& array, int array_offset) const
{
    array[0 + array_offset] = { 0, 1, -1 };
}
std::shared_ptr<Geometry> SphereGeometry::transoform(glm::mat4& transformation_matrix) const
{
    auto sphere = std::make_shared<SphereGeometry>(_radius);
    sphere->_position = transformation_matrix * glm::vec4f(_position, 1.0f);
    return sphere;
}
}
#include "sphere.h"
#include "../header/enum.h"

namespace rtx {
SphereGeometry::SphereGeometry(float radius)
    : Geometry()
{
    _radius = radius;
    _center = { 0.0f, 0.0f, 0.0f, 1.0f };
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
    array[0 + offset] = { _center.x, _center.y, _center.z, 1.0f };
    array[1 + offset] = { _radius, -1.0f, -1.0f, -1.0f };
}
void SphereGeometry::serialize_faces(rtx::array<rtxFaceVertexIndex>& array, int array_offset) const
{
    array[0 + array_offset] = { 0, 1, -1 };
}
std::shared_ptr<Geometry> SphereGeometry::transoform(glm::mat4& transformation_matrix) const
{
    auto sphere = std::make_shared<SphereGeometry>(_radius);
    sphere->_center = transformation_matrix * _center;
    return sphere;
}
}
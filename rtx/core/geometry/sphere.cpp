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
void SphereGeometry::pack_vertices(float*& buffer, int start, glm::mat4& transformation_matrix) const
{
}
}
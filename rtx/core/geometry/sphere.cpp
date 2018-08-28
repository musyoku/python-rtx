#include "sphere.h"

namespace rtx {
SphereGeometry::SphereGeometry(float radius)
{
    _radius = radius;
    _center = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
}

int SphereGeometry::type()
{
    return RTX_GEOMETRY_TYPE_SPHERE;
}
}
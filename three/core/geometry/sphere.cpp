#include "sphere.h"

namespace three {
SphereGeometry::SphereGeometry(float radius)
{
    _radius = radius;
    _center = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
}

GeometryType SphereGeometry::type()
{
    return GeometryTypeSphere;
}
}
#include "sphere.h"

namespace three {
SphereGeometry::SphereGeometry(float radius)
{
    _radius = radius;
}

GeometryType SphereGeometry::type()
{
    return GeometryTypeSphere;
}
}
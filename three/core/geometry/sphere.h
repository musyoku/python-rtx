#pragma once
#include "../class/geometry.h"

namespace three {
class SphereGeometry : public Geometry {
public:
    float _radius;
    SphereGeometry(float radius);
    GeometryType type();
};
}
#pragma once
#include "../class/geometry.h"

namespace three {
class SphereGeometry : public Geometry {
public:
    SphereGeometry(float radius);
};
}
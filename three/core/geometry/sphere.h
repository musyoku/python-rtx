#pragma once
#include "../class/geometry.h"
#include <glm/glm.hpp>

namespace three {
class SphereGeometry : public Geometry {
public:
    float _radius;
    glm::vec4 _center;
    SphereGeometry(float radius);
    GeometryType type() override;
};
}
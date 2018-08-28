#pragma once
#include "../class/geometry.h"
#include <glm/glm.hpp>

namespace rtx {
class SphereGeometry : public Geometry {
public:
    float _radius;
    glm::vec4 _center;
    SphereGeometry(float radius);
    int type() override;
};
}
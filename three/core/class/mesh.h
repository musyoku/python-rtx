#pragma once
#include "geometry.h"
#include "material.h"
#include <memory>

namespace three {
class Mesh {
public:
    Mesh(std::shared_ptr<Geometry> geometry, std::shared_ptr<Material> material);
};
}
#pragma once
#include "geometry.h"
#include "material.h"
#include <memory>

namespace three {
class Mesh {
public:
    std::shared_ptr<Geometry> _geometry;
    std::shared_ptr<Material> _material;
    Mesh(std::shared_ptr<Geometry> geometry, std::shared_ptr<Material> material);
};
}
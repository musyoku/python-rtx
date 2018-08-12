#pragma once
#include "geometry.h"
#include "material.h"

namespace three {
class Mesh {
public:
    Mesh(Geometry* geometry, Material* material);
};
}
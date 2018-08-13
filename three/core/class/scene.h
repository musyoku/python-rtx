#pragma once
#include "mesh.h"
#include <memory>
#include <vector>

namespace three {
class Scene {
public:
    std::vector<std::shared_ptr<Mesh>> _mesh_array;
    void add(std::shared_ptr<Mesh> mesh);
};
}
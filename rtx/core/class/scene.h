#pragma once
#include "mesh.h"
#include "light.h"
#include <memory>
#include <vector>

namespace rtx {
class Scene {
private:
    bool _updated;

public:
    std::vector<std::shared_ptr<Mesh>> _mesh_array;
    std::vector<std::shared_ptr<Light>> _light_array;
    void add(std::shared_ptr<Mesh> mesh);
    void add(std::shared_ptr<Light> light);
    bool updated();
    void set_updated(bool updated);
    int num_triangles();
};
}
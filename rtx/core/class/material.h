#pragma once
#include "../header/struct.h"
#include <memory>
#include <vector>

namespace rtx {
class Material {
public:
    virtual int type() const = 0;
    virtual int attribute_bytes() const = 0;
};

class LayeredMaterial {
public:
    std::vector<std::shared_ptr<Material>> _material_array;
    LayeredMaterial(std::shared_ptr<Material> material);
    LayeredMaterial(std::shared_ptr<Material> top, std::shared_ptr<Material> bottom);
    LayeredMaterial(std::shared_ptr<Material> top, std::shared_ptr<Material> middle, std::shared_ptr<Material> bottom);
    int attribute_bytes();
    int num_layers();
    RTXLayeredMaterialTypes types();
};
}
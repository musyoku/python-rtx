#pragma once
#include <memory>
#include <vector>

namespace rtx {
class Material {
    virtual int type() const = 0;
};
class LayeredMaterial {
public:
    std::vector<std::shared_ptr<Material>> _material_array;
    LayeredMaterial(std::shared_ptr<Material> material);
    LayeredMaterial(std::shared_ptr<Material> top, std::shared_ptr<Material> bottom);
    LayeredMaterial(std::shared_ptr<Material> top, std::shared_ptr<Material> middle, std::shared_ptr<Material> bottom);
};
}
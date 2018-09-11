#include "material.h"

namespace rtx {
LayeredMaterial::LayeredMaterial(std::shared_ptr<Material> material)
{
    _material_array.push_back(material);
}
LayeredMaterial::LayeredMaterial(std::shared_ptr<Material> top, std::shared_ptr<Material> bottom)
{
    _material_array.push_back(top);
    _material_array.push_back(bottom);
}
LayeredMaterial::LayeredMaterial(std::shared_ptr<Material> top, std::shared_ptr<Material> middle, std::shared_ptr<Material> bottom)
{
    _material_array.push_back(top);
    _material_array.push_back(middle);
    _material_array.push_back(bottom);
}
}
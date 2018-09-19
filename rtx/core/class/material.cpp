#include "material.h"
#include "../header/enum.h"

namespace rtx {
LayeredMaterial::LayeredMaterial(std::shared_ptr<Material> material)
{
    _material_array.push_back(material);
}
LayeredMaterial::LayeredMaterial(std::shared_ptr<Material> outside, std::shared_ptr<Material> inside)
{
    _material_array.push_back(outside);
    _material_array.push_back(inside);
}
LayeredMaterial::LayeredMaterial(std::shared_ptr<Material> top, std::shared_ptr<Material> middle, std::shared_ptr<Material> inside)
{
    _material_array.push_back(top);
    _material_array.push_back(middle);
    _material_array.push_back(inside);
}
int LayeredMaterial::attribute_bytes()
{
    int bytes = 0;
    for (auto& material : _material_array) {
        bytes += material->attribute_bytes();
    }
    return bytes;
}
int LayeredMaterial::num_layers()
{
    return _material_array.size();
}
rtxLayeredMaterialTypes LayeredMaterial::types()
{
    rtxLayeredMaterialTypes types;
    types.outside = -1;
    types.middle = -1;
    types.inside = -1;

    if (_material_array.size() == 1) {
        types.outside = _material_array[0]->type();
        return types;
    }

    if (_material_array.size() == 2) {
        types.outside = _material_array[0]->type();
        types.middle = _material_array[1]->type();
        return types;
    }

    if (_material_array.size() == 3) {
        types.outside = _material_array[0]->type();
        types.middle = _material_array[1]->type();
        types.inside = _material_array[2]->type();
        return types;
    }

    throw std::runtime_error("invalid layered material");
}
void LayeredMaterial::serialize_attributes(rtx::array<rtxMaterialAttributeByte>& array, int offset) const
{
    for (auto& material : _material_array) {
        assert(offset < array.size());
        material->serialize_attributes(array, offset);
        offset += material->attribute_bytes() / sizeof(rtxMaterialAttributeByte);
    }
}
bool LayeredMaterial::is_emissive()
{
    return _material_array[0]->type() == RTXMaterialTypeEmissive;
}
}
#include "emissive.h"
#include "../header/enum.h"
#include "../header/struct.h"
#include <cstring>

namespace rtx {
EmissiveMaterial::EmissiveMaterial(float intensity)
{
    _intensity = intensity;
    _visible = true;
}
EmissiveMaterial::EmissiveMaterial(float intensity, bool visible)
{
    _intensity = intensity;
    _visible = visible;
}
float EmissiveMaterial::intensity() const
{
    return _intensity;
}
int EmissiveMaterial::type() const
{
    return RTXMaterialTypeEmissive;
}
int EmissiveMaterial::attribute_bytes() const
{
    return sizeof(rtxEmissiveMaterialAttribute);
}
void EmissiveMaterial::serialize_attributes(rtx::array<rtxMaterialAttributeByte>& array, int offset) const
{
    rtxEmissiveMaterialAttribute attr;
    attr.intensity = _intensity;
    attr.visible = _visible;
    rtxMaterialAttributeByte* pointer = array.data();
    std::memcpy(&pointer[offset], &attr, sizeof(rtxEmissiveMaterialAttribute));
}
}
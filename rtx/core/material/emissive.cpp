#include "emissive.h"
#include "../header/enum.h"
#include "../header/struct.h"
#include <cstring>

namespace rtx {
EmissiveMaterial::EmissiveMaterial(float brightness)
{
    _brightness = brightness;
    _visible = true;
}
EmissiveMaterial::EmissiveMaterial(float brightness, bool visible)
{
    _brightness = brightness;
    _visible = visible;
}
float EmissiveMaterial::brightness() const
{
    return _brightness;
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
    attr.brightness = _brightness;
    attr.visible = _visible;
    rtxMaterialAttributeByte* pointer = array.data();
    std::memcpy(&pointer[offset], &attr, sizeof(rtxEmissiveMaterialAttribute));
}
}
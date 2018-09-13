#include "emissive.h"
#include "../header/enum.h"
#include "../header/struct.h"
#include <cstring>

namespace rtx {
EmissiveMaterial::EmissiveMaterial(float brightness)
{
    _brightness = brightness;
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
    return sizeof(RTXEmissiveMaterialAttribute);
}
void EmissiveMaterial::serialize_attributes(rtx::array<RTXMaterialAttributeByte>& array, int offset) const
{
    RTXEmissiveMaterialAttribute attr;
    attr.brightness = _brightness;
    RTXMaterialAttributeByte* pointer = array.data();
    std::memcpy(&pointer[offset], &attr, sizeof(RTXEmissiveMaterialAttribute));
}
}
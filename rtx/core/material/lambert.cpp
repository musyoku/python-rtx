#include "lambert.h"
#include "../header/enum.h"
#include "../header/struct.h"
#include <cstring>

namespace rtx {
LambertMaterial::LambertMaterial(float albedo)
{
    _albedo = albedo;
}
int LambertMaterial::type() const
{
    return RTXMaterialTypeLambert;
}
int LambertMaterial::attribute_bytes() const
{
    return sizeof(RTXLambertMaterialAttribute);
}
void LambertMaterial::serialize_attributes(rtx::array<RTXMaterialAttributeByte>& array, int offset) const
{
    RTXLambertMaterialAttribute attr;
    attr.albedo = _albedo;
    RTXMaterialAttributeByte* pointer = array.data();
    std::memcpy(&pointer[offset], &attr, sizeof(RTXLambertMaterialAttribute));
}
}
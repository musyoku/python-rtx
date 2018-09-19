#include "lambert.h"
#include "../header/enum.h"
#include "../header/struct.h"
#include <cstring>

namespace rtx {
LambertMaterial::LambertMaterial(float albedo)
{
    _albedo = albedo;
}
float LambertMaterial::albedo() const
{
    return _albedo;
}
int LambertMaterial::type() const
{
    return RTXMaterialTypeLambert;
}
int LambertMaterial::attribute_bytes() const
{
    return sizeof(rtxLambertMaterialAttribute);
}
void LambertMaterial::serialize_attributes(rtx::array<rtxMaterialAttributeByte>& array, int offset) const
{
    rtxLambertMaterialAttribute attr;
    attr.albedo = _albedo;
    rtxMaterialAttributeByte* pointer = array.data();
    std::memcpy(&pointer[offset], &attr, sizeof(rtxLambertMaterialAttribute));
}
}
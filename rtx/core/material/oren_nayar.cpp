#include "oren_nayar.h"
#include "../header/enum.h"
#include "../header/struct.h"
#include <cstring>

namespace rtx {
OrenNayarMaterial::OrenNayarMaterial(float albedo, float roughness)
{
    _albedo = albedo;
    _roughness = roughness;
}
float OrenNayarMaterial::albedo() const
{
    return _albedo;
}
float OrenNayarMaterial::roughness() const
{
    return _roughness;
}
int OrenNayarMaterial::type() const
{
    return RTXMaterialTypeOrenNayar;
}
int OrenNayarMaterial::attribute_bytes() const
{
    return sizeof(rtxOrenNayarMaterialAttribute);
}
void OrenNayarMaterial::serialize_attributes(rtx::array<rtxMaterialAttributeByte>& array, int offset) const
{
    rtxOrenNayarMaterialAttribute attr;
    attr.albedo = _albedo;
    attr.roughness = _roughness;
    rtxMaterialAttributeByte* pointer = array.data();
    std::memcpy(&pointer[offset], &attr, sizeof(rtxOrenNayarMaterialAttribute));
}
}
#include "lambert.h"
#include "../header/enum.h"
#include "../header/struct.h"

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
}
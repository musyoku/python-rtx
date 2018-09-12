#include "lambert.h"
#include "../header/enum.h"

namespace rtx {
LambertMaterial::LambertMaterial(float albedo)
{
    _albedo = albedo;
}
int LambertMaterial::type() const
{
    return RTXMaterialTypeLambert;
}
}
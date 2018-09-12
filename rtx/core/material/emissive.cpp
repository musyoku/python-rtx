#include "emissive.h"
#include "../header/enum.h"
#include "../header/struct.h"

namespace rtx {
EmissiveMaterial::EmissiveMaterial(float brightness)
{
    _brightness = brightness;
}
int EmissiveMaterial::type() const
{
    return RTXMaterialTypeEmissive;
}
int EmissiveMaterial::attribute_bytes() const
{
    return sizeof(RTXEmissiveMaterialAttribute);
}
}
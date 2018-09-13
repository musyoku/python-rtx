#pragma once
#include "../class/material.h"
#include "../header/array.h"

namespace rtx {
class LambertMaterial : public Material {
private:
    float _albedo;

public:
    LambertMaterial(float albedo);
    int type() const override;
    int attribute_bytes() const override;
    void serialize_attributes(rtx::array<RTXMaterialAttributeByte>& array, int offset) const override;
};
}
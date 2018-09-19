#pragma once
#include "../class/material.h"
#include "../header/array.h"

namespace rtx {
class LambertMaterial : public Material {
private:
    float _albedo;

public:
    LambertMaterial(float albedo);
    float albedo() const;
    int type() const override;
    int attribute_bytes() const override;
    void serialize_attributes(rtx::array<rtxMaterialAttributeByte>& array, int offset) const override;
};
}
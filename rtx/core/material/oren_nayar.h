#pragma once
#include "../class/material.h"
#include "../header/array.h"

namespace rtx {
class OrenNayarMaterial : public Material {
private:
    float _albedo;
    float _roughness;

public:
    OrenNayarMaterial(float albedo, float roughness);
    float albedo() const;
    float roughness() const;
    int type() const override;
    int attribute_bytes() const override;
    void serialize_attributes(rtx::array<rtxMaterialAttributeByte>& array, int offset) const override;
};
}
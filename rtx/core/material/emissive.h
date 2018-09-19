#pragma once
#include "../class/material.h"
#include "../header/array.h"

namespace rtx {
class EmissiveMaterial : public Material {
private:
    float _brightness;

public:
    EmissiveMaterial(float brightness);
    float brightness() const;
    int type() const override;
    int attribute_bytes() const override;
    void serialize_attributes(rtx::array<rtxMaterialAttributeByte>& array, int offset) const override;
};
}
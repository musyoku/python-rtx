#pragma once
#include "../class/material.h"
#include "../header/array.h"

namespace rtx {
class EmissiveMaterial : public Material {
private:
    float _brightness;

public:
    EmissiveMaterial(float brightness);
    int type() const override;
    int attribute_bytes() const override;
    void serialize_attributes(rtx::array<RTXMaterialAttributeByte>& array, int offset) const override;
};
}
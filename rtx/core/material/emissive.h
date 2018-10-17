#pragma once
#include "../class/material.h"
#include "../header/array.h"

namespace rtx {
class EmissiveMaterial : public Material {
private:
    float _brightness;
    bool _visible;

public:
    EmissiveMaterial(float brightness);
    EmissiveMaterial(float brightness, bool visible);
    float brightness() const;
    int type() const override;
    int attribute_bytes() const override;
    void serialize_attributes(rtx::array<rtxMaterialAttributeByte>& array, int offset) const override;
};
}
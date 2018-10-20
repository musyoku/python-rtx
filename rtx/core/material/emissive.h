#pragma once
#include "../class/material.h"
#include "../header/array.h"

namespace rtx {
class EmissiveMaterial : public Material {
private:
    float _intensity;
    bool _visible;

public:
    EmissiveMaterial(float intensity);
    EmissiveMaterial(float intensity, bool visible);
    float intensity() const;
    int type() const override;
    int attribute_bytes() const override;
    void serialize_attributes(rtx::array<rtxMaterialAttributeByte>& array, int offset) const override;
};
}
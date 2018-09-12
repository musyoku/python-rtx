#pragma once
#include "../class/material.h"

namespace rtx {
class EmissiveMaterial : public Material {
private:
    float _brightness;

public:
    EmissiveMaterial(float brightness);
    int type() const override;
    int attribute_bytes() const override;
};
}
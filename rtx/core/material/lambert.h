#pragma once
#include "../class/material.h"
#include "../header/glm.h"
#include <pybind11/pybind11.h>

namespace rtx {
class LambertMaterial : public Material {
private:
    float _albedo;

public:
    LambertMaterial(float albedo);
    int type() const override;
    int attribute_bytes() const override;
};
}
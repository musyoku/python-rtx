#pragma once
#include "../class/material.h"
#include "../header/glm.h"
#include <pybind11/pybind11.h>

namespace rtx {
class EmissiveMaterial : public Material {
public:
    // color: [0, 1]
    EmissiveMaterial(pybind11::tuple color, float diffuse_reflectance);
    EmissiveMaterial(float (&color)[3], float diffuse_reflectance);
    int type() const override;
};
}
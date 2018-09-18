#pragma once
#include "../class/mapping.h"
#include "../header/array.h"
#include "../header/glm.h"
#include <pybind11/pybind11.h>

namespace rtx {
class SolidColorMapping : public Mapping {
protected:
    glm::vec4f _color;

public:
    SolidColorMapping(pybind11::tuple color);
    SolidColorMapping(float (&color)[3]);
    void set_color(pybind11::tuple color);
    void set_color(float (&color)[3]);
    int type() const override;
    glm::vec4f color();
};
}
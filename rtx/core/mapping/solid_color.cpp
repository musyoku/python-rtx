#include "solid_color.h"
#include "../header/enum.h"

namespace rtx {
SolidColorMapping::SolidColorMapping(pybind11::tuple color)
{
    set_color(color);
}
SolidColorMapping::SolidColorMapping(float (&color)[3])
{
    set_color(color);
}
void SolidColorMapping::set_color(pybind11::tuple color)
{
    _color[0] = color[0].cast<float>();
    _color[1] = color[1].cast<float>();
    _color[2] = color[2].cast<float>();
}
void SolidColorMapping::set_color(float (&color)[3])
{
    _color[0] = color[0];
    _color[1] = color[1];
    _color[2] = color[2];
}
int SolidColorMapping::type() const
{
    return RTXMappingTypeSolidColor;
}
glm::vec4f SolidColorMapping::color()
{
    return _color;
}
}
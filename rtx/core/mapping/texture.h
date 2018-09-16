#pragma once
#include "../class/mapping.h"
#include "../header/array.h"
#include "../header/struct.h"
#include <pybind11/numpy.h>

namespace rtx {
class TextureMapping : public Mapping {
private:
    rtx::array<RTXPixel> _texture;
    int _width;
    int _height;

public:
    TextureMapping(pybind11::array_t<float, pybind11::array::c_style> texture);
    int type() const override;
    int bytes();
    int width();
    int height();
    RTXPixel* pointer();
};
}
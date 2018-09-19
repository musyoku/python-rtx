#pragma once
#include "../class/mapping.h"
#include "../header/array.h"
#include "../header/struct.h"
#include <pybind11/numpy.h>

namespace rtx {
class TextureMapping : public Mapping {
private:
    rtx::array<rtxRGBAPixel> _texture;
    rtx::array<rtxUVCoordinate> _uv_coordinates;
    int _width;
    int _height;

public:
    TextureMapping(
        pybind11::array_t<float, pybind11::array::c_style> texture,
        pybind11::array_t<float, pybind11::array::c_style> uv_coordinates);
    int bytes();
    int width();
    int height();
    int num_uv_coordinates();
    int type() const override;
    void serialize_uv_coordinates(rtx::array<rtxUVCoordinate>& array, int offset) const;
    rtxRGBAPixel* data();
};
}
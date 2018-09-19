#include "texture.h"
#include "../header/enum.h"

namespace rtx {
TextureMapping::TextureMapping(
    pybind11::array_t<float, pybind11::array::c_style> np_texture,
    pybind11::array_t<float, pybind11::array::c_style> np_uv_coordinates)
{
    if (np_texture.ndim() != 3) {
        throw std::runtime_error("(np_texture.ndim() != 3) -> false");
    }
    if (np_uv_coordinates.ndim() != 2) {
        throw std::runtime_error("(np_uv_coordinates.ndim() != 2) -> false");
    }
    _height = np_texture.shape(0);
    _width = np_texture.shape(1);
    int channels = np_texture.shape(2);
    if (channels != 1 && channels != 3) {
        throw std::runtime_error("(channels != 1 && channels != 3) -> false");
    }

    auto texture = np_texture.mutable_unchecked<3>();
    _texture = rtx::array<rtxRGBAPixel>(_height * _width);
    for (int h = 0; h < _height; h++) {
        for (int w = 0; w < _width; w++) {
            _texture[h * _width + w] = rtxRGBAPixel({ texture(h, w, 0),
                texture(h, w, 1),
                texture(h, w, 2),
                1.0f });
        }
    }

    int num_attributes = np_uv_coordinates.shape(0);
    auto uv_coordinates = np_uv_coordinates.mutable_unchecked<2>();
    _uv_coordinates = rtx::array<rtxUVCoordinate>(num_attributes);
    for (int vertex_index = 0; vertex_index < num_attributes; vertex_index++) {
        _uv_coordinates[vertex_index] = rtxUVCoordinate({
            uv_coordinates(vertex_index, 0),
            uv_coordinates(vertex_index, 1),
        });
    }
}
int TextureMapping::width()
{
    return _width;
}
int TextureMapping::height()
{
    return _height;
}
int TextureMapping::type() const
{
    return RTXMappingTypeTexture;
}
int TextureMapping::bytes()
{
    return sizeof(rtxRGBAPixel) * _texture.size();
}
int TextureMapping::num_uv_coordinates()
{
    return _uv_coordinates.size();
}
rtxRGBAPixel* TextureMapping::data()
{
    return _texture.data();
}
void TextureMapping::serialize_uv_coordinates(rtx::array<rtxUVCoordinate>& array, int offset) const
{
    for (int n = 0; n < _uv_coordinates.size(); n++) {
        array[n + offset] = _uv_coordinates[n];
    }
}
}
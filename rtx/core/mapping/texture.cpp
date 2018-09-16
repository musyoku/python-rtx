#include "texture.h"
#include "../header/enum.h"

namespace rtx {
TextureMapping::TextureMapping(pybind11::array_t<float, pybind11::array::c_style> np_texture)
{
    if (np_texture.ndim() != 3) {
        throw std::runtime_error("(np_texture.ndim() != 3) -> false");
    }
    _height = np_texture.shape(0);
    _width = np_texture.shape(1);
    int channels = np_texture.shape(2);
    if (channels != 1 && channels != 3) {
        throw std::runtime_error("(channels != 1 && channels != 3) -> false");
    }

    auto texture = np_texture.mutable_unchecked<3>();
    _texture = rtx::array<RTXPixel>(_height * _width);
    for (int h = 0; h < _height; h++) {
        for (int w = 0; w < _width; w++) {
            _texture[h * _width + w] = RTXPixel({ texture(h, w, 0),
                texture(h, w, 1),
                texture(h, w, 2),
                1.0f });
        }
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
    return sizeof(RTXPixel) * _texture.size();
}
RTXPixel* TextureMapping::pointer()
{
    return _texture.data();
}
}
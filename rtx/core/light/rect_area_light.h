#pragma once
#include "../class/light.h"
namespace rtx {

class RectAreaLight : public Light {
private:
    int _width;
    int _height;

public:
    RectAreaLight(int width, int height);
}
}
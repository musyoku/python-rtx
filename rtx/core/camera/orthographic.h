#pragma once
#include "../class/camera.h"
#include <pybind11/pybind11.h>

namespace rtx {
class OrthographicCamera : public Camera {
public:
    OrthographicCamera(pybind11::tuple eye, pybind11::tuple center, pybind11::tuple up);
    RTXCameraType type() const override;
};
}
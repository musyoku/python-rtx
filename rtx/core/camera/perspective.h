#pragma once
#include "../class/camera.h"
#include <pybind11/pybind11.h>

namespace rtx {
class PerspectiveCamera : public Camera {
public:
    float _fov_rad;
    float _aspect_ratio;
    float _z_near;
    float _z_far;
    PerspectiveCamera();
    PerspectiveCamera(pybind11::tuple eye, pybind11::tuple center, pybind11::tuple up,
        float fov_rad, float aspect_ratio, float z_near, float z_far);
    PerspectiveCamera(float (&eye)[3], float (&center)[3], float (&up)[3],
        float fov_rad, float aspect_ratio, float z_near, float z_far);
    float fov_rad();
    void set_fov_rad(float fov_rad);
    RTXCameraType type() const override;
};
}
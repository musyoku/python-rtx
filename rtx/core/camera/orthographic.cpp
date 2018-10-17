#include "orthographic.h"

namespace rtx {
namespace py = pybind11;
OrthographicCamera::OrthographicCamera()
{
    float eye[3] = { 0, 0, 1 };
    float center[3] = { 0, 0, 0 };
    float up[3] = { 0, 1, 0 };
    look_at(eye, center, up);
}
OrthographicCamera::OrthographicCamera(py::tuple eye, py::tuple center, py::tuple up)
{
    look_at(eye, center, up);
}
RTXCameraType OrthographicCamera::type() const
{
    return RTXCameraTypeOrthographic;
}
}
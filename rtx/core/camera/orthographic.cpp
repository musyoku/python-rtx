#include "orthographic.h"

namespace rtx {
namespace py = pybind11;
OrthographicCamera::OrthographicCamera(py::tuple eye, py::tuple center, py::tuple up)
{
    look_at(eye, center, up);
}
RTXCameraType OrthographicCamera::type() const
{
    return RTXCameraTypeOrthographic;
}
}
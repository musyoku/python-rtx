#pragma once
#include "../class/geometry.h"
#include <glm/glm.hpp>
#include <memory>
#include <pybind11/numpy.h>
#include <vector>

namespace rtx {
class StandardGeometry : public Geometry {
public:
    std::vector<glm::vec<3, int>> _face_vertex_indices_array;
    std::vector<glm::vec4> _vertex_array;
    StandardGeometry();
    StandardGeometry(pybind11::array_t<int, pybind11::array::c_style> face_vertex_indeces, pybind11::array_t<float, pybind11::array::c_style> vertices);
    GeometryType type() override;
};
}
#include "light.h"
#include <cassert>
#include <glm/gtc/matrix_transform.hpp>

namespace rtx {
namespace py = pybind11;
void Light::set_position(py::tuple position)
{
    _position[0] = position[0].cast<float>();
    _position[1] = position[1].cast<float>();
    _position[2] = position[2].cast<float>();
    update_model_matrix();
}
void Light::set_rotation(py::tuple rotation_rad)
{
    _rotation_rad[0] = rotation_rad[0].cast<float>();
    _rotation_rad[1] = rotation_rad[1].cast<float>();
    _rotation_rad[2] = rotation_rad[2].cast<float>();
    update_model_matrix();
}
void Light::update_model_matrix()
{
    _model_matrix = glm::mat4(1.0);
    _model_matrix = glm::translate(_model_matrix, _position);
    _model_matrix = glm::rotate(_model_matrix, _rotation_rad[0], glm::vec3(1.0f, 0.0f, 0.0f));
    _model_matrix = glm::rotate(_model_matrix, _rotation_rad[1], glm::vec3(0.0f, 1.0f, 0.0f));
    _model_matrix = glm::rotate(_model_matrix, _rotation_rad[2], glm::vec3(0.0f, 0.0f, 1.0f));
}
bool Light::is_light() const
{
    return true;
}
float Light::brightness() const
{
    return _brightness;
}
}
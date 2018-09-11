#include "geometry.h"

namespace rtx {
void Mesh::set_scale(py::tuple scale)
{
    _scale[0] = scale[0].cast<float>();
    _scale[1] = scale[1].cast<float>();
    _scale[2] = scale[2].cast<float>();
    update_model_matrix();
}
void Mesh::set_scale(float (&scale)[3])
{
    _scale[0] = scale[0];
    _scale[1] = scale[1];
    _scale[2] = scale[2];
    update_model_matrix();
}
void Mesh::set_position(py::tuple position)
{
    _position[0] = position[0].cast<float>();
    _position[1] = position[1].cast<float>();
    _position[2] = position[2].cast<float>();
    update_model_matrix();
}
void Mesh::set_position(float (&position)[3])
{
    _position[0] = position[0];
    _position[1] = position[1];
    _position[2] = position[2];
    update_model_matrix();
}
void Mesh::set_rotation(py::tuple rotation_rad)
{
    _rotation_rad[0] = rotation_rad[0].cast<float>();
    _rotation_rad[1] = rotation_rad[1].cast<float>();
    _rotation_rad[2] = rotation_rad[2].cast<float>();
    update_model_matrix();
}
void Mesh::set_rotation(float (&rotation)[3])
{
    _rotation[0] = rotation[0];
    _rotation[1] = rotation[1];
    _rotation[2] = rotation[2];
    update_model_matrix();
}
void Mesh::update_model_matrix()
{
    _model_matrix = glm::mat4(1.0);
    _model_matrix = glm::translate(_model_matrix, _position);
    _model_matrix = glm::rotate(_model_matrix, _rotation_rad[0], glm::vec3(1.0f, 0.0f, 0.0f));
    _model_matrix = glm::rotate(_model_matrix, _rotation_rad[1], glm::vec3(0.0f, 1.0f, 0.0f));
    _model_matrix = glm::rotate(_model_matrix, _rotation_rad[2], glm::vec3(0.0f, 0.0f, 1.0f));
    _model_matrix = glm::scale(_model_matrix, _scale);
}
}
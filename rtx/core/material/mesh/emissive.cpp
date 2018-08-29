#include "emissive.h"

namespace rtx {
MeshEmissiveMaterial::MeshEmissiveMaterial(pybind11::tuple color)
{
    _color.r = color[0].cast<float>();
    _color.g = color[1].cast<float>();
    _color.b = color[2].cast<float>();
}

MeshEmissiveMaterial::MeshEmissiveMaterial(float (&color)[3])
{
    _color.r = color[0];
    _color.g = color[1];
    _color.b = color[2];
}
glm::vec3f MeshEmissiveMaterial::emit_color() const
{
    return _color;
}
glm::vec3f MeshEmissiveMaterial::reflect_color(glm::vec3f& input_color) const
{
    throw std::runtime_error("Not implemented");
};
glm::vec3f MeshEmissiveMaterial::reflect_ray(glm::vec3f& diffuse_vec, glm::vec3f& specular_vec) const
{
    throw std::runtime_error("Not implemented");
};
glm::vec3f MeshEmissiveMaterial::color() const
{
    return _color;
}
int MeshEmissiveMaterial::type() const
{
    return RTX_MATERIAL_TYPE_EMISSIVE;
}
}
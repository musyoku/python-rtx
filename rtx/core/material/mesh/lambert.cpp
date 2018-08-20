#include "lambert.h"

namespace rtx {
MeshLambertMaterial::MeshLambertMaterial(pybind11::tuple color,
    float diffuse_reflectance)
{
    _color.r = color[0].cast<float>();
    _color.g = color[1].cast<float>();
    _color.b = color[2].cast<float>();
    _diffuse_reflectance = diffuse_reflectance;
}

MeshLambertMaterial::MeshLambertMaterial(float (&color)[3], float diffuse_reflectance)
{
    _color.r = color[0];
    _color.g = color[1];
    _color.b = color[2];
    _diffuse_reflectance = diffuse_reflectance;
}
glm::vec3 MeshLambertMaterial::reflect_color(glm::vec3& input_color) const
{
    return _diffuse_reflectance * _color * input_color;
}

glm::vec3 MeshLambertMaterial::reflect_ray(glm::vec3& diffuse_vec, glm::vec3& specular_vec) const
{
    return diffuse_vec;
}
glm::vec3 MeshLambertMaterial::emit_color() const
{
    throw std::runtime_error("Not implemented");
};
MaterialType MeshLambertMaterial::type() const
{
    return MaterialTypeLambert;
}
}
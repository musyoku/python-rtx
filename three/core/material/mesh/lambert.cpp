#include "lambert.h"

namespace three {
MeshLambertMaterial::MeshLambertMaterial(pybind11::tuple color,
    float diffuse_reflectance)
{
    _color.r = color[0].cast<float>();
    _color.g = color[1].cast<float>();
    _color.b = color[2].cast<float>();

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
MaterialType MeshLambertMaterial::type() const
{
    return MaterialTypeLambert;
}
}
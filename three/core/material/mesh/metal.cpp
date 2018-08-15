#include "metal.h"

namespace three {
MeshMetalMaterial::MeshMetalMaterial(float roughness, float specular_reflectance)
{
    _roughness = roughness;
    _specular_reflectance = specular_reflectance;
}
glm::vec3 MeshMetalMaterial::reflect_color(glm::vec3& input_color) const
{
    return _specular_reflectance * input_color;
}

glm::vec3 MeshMetalMaterial::reflect_ray(glm::vec3& diffuse_vec, glm::vec3& specular_vec) const
{
    return (1.0f - _roughness) * diffuse_vec + _roughness * specular_vec;
}
MaterialType MeshMetalMaterial::type() const
{
    return MaterialTypeMetal;
}
}
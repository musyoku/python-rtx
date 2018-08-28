#include "metal.h"

namespace rtx {
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
glm::vec3 MeshMetalMaterial::emit_color() const
{
    throw std::runtime_error("Not implemented");
};
glm::vec3 MeshMetalMaterial::color() const
{
    throw std::runtime_error("Not implemented");
};
int MeshMetalMaterial::type() const
{
    return RTX_MATERIAL_TYPE_METAL;
}
}
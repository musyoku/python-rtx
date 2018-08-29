#include "metal.h"

namespace rtx {
MeshMetalMaterial::MeshMetalMaterial(float roughness, float specular_reflectance)
{
    _roughness = roughness;
    _specular_reflectance = specular_reflectance;
}
glm::vec3f MeshMetalMaterial::reflect_color(glm::vec3f& input_color) const
{
    return _specular_reflectance * input_color;
}

glm::vec3f MeshMetalMaterial::reflect_ray(glm::vec3f& diffuse_vec, glm::vec3f& specular_vec) const
{
    return (1.0f - _roughness) * diffuse_vec + _roughness * specular_vec;
}
glm::vec3f MeshMetalMaterial::emit_color() const
{
    throw std::runtime_error("Not implemented");
};
glm::vec3f MeshMetalMaterial::color() const
{
    throw std::runtime_error("Not implemented");
};
int MeshMetalMaterial::type() const
{
    return RTX_MATERIAL_TYPE_METAL;
}
}
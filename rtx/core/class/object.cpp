#include "object.h"
#include "../header/enum.h"

namespace rtx {
namespace py = pybind11;
Object::Object(std::shared_ptr<Geometry> geometry, std::shared_ptr<Material> material, std::shared_ptr<Mapping> mapping)
{
    if (mapping->type() == RTXMappingTypeTexture) {
        switch (geometry->type()) {
        case RTXGeometryTypeCone:
        case RTXGeometryTypeCylinder:
        case RTXGeometryTypeSphere:
            throw std::runtime_error("UV Mapping for Cone, Cylinder, and Sphere is currently not supported.");
            break;
        default:
            break;
        }
    }
    set_geometry(geometry);
    set_material(material);
    set_mapping(mapping);
}
Object::Object(std::shared_ptr<Geometry> geometry, std::shared_ptr<LayeredMaterial> material, std::shared_ptr<Mapping> mapping)
{
    if (mapping->type() == RTXMappingTypeTexture) {
        switch (geometry->type()) {
        case RTXGeometryTypeCone:
        case RTXGeometryTypeCylinder:
        case RTXGeometryTypeSphere:
            throw std::runtime_error("UV Mapping for Cone, Cylinder, and Sphere is currently not supported.");
            break;
        default:
            break;
        }
    }
    set_geometry(geometry);
    set_material(material);
    set_mapping(mapping);
}
void Object::set_geometry(std::shared_ptr<Geometry> geometry)
{
    _geometry = geometry;
}
void Object::set_material(std::shared_ptr<Material> material)
{
    _material = std::make_shared<LayeredMaterial>(material);
}
void Object::set_material(std::shared_ptr<LayeredMaterial> material)
{
    _material = material;
}
void Object::set_mapping(std::shared_ptr<Mapping> mapping)
{
    _mapping = mapping;
}
std::shared_ptr<Geometry>& Object::geometry()
{
    return _geometry;
}
std::shared_ptr<LayeredMaterial>& Object::material()
{
    return _material;
}
std::shared_ptr<Mapping>& Object::mapping()
{
    return _mapping;
}
void ObjectGroup::add(std::shared_ptr<Object> object)
{
    _object_array.push_back(object);
}
}
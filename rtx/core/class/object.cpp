#include "object.h"

namespace rtx {
namespace py = pybind11;
Object::Object(std::shared_ptr<Geometry> geometry, std::shared_ptr<Material> material, std::shared_ptr<Mapping> mapping)
{
    set_geometry(geometry);
    set_material(material);
    set_mapping(mapping);
}
Object::Object(std::shared_ptr<Geometry> geometry, std::shared_ptr<LayeredMaterial> material, std::shared_ptr<Mapping> mapping)
{
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
}
#pragma once
#include "../header/glm.h"
#include "geometry.h"
#include "mapping.h"
#include "material.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace rtx {
class Object {
private:
    std::shared_ptr<Geometry> _geometry;
    std::shared_ptr<LayeredMaterial> _material;
    std::shared_ptr<Mapping> _mapping;

public:
    Object(std::shared_ptr<Geometry> geometry, std::shared_ptr<Material> material, std::shared_ptr<Mapping> mapping);
    Object(std::shared_ptr<Geometry> geometry, std::shared_ptr<LayeredMaterial> material, std::shared_ptr<Mapping> mapping);
    void set_geometry(std::shared_ptr<Geometry> geometry);
    void set_material(std::shared_ptr<LayeredMaterial> material);
    void set_material(std::shared_ptr<Material> material);
    void set_mapping(std::shared_ptr<Mapping> mapping);
    std::shared_ptr<Geometry>& geometry();
    std::shared_ptr<LayeredMaterial>& material();
    std::shared_ptr<Mapping>& mapping();
};
}
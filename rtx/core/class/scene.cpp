#include "scene.h"

namespace rtx {
Scene::Scene(pybind11::tuple ambient_color)
{
    _ambient_color = {
        ambient_color[0].cast<float>(),
        ambient_color[1].cast<float>(),
        ambient_color[2].cast<float>(),
    };
}
void Scene::add(std::shared_ptr<Object> object)
{
    _object_array.emplace_back(object);
    _updated = true;
}
bool Scene::updated()
{
    if (_updated) {
        return true;
    }
    for (auto& object : _object_array) {
        if (object->geometry()->updated()) {
            return true;
        }
    }
    return false;
}
void Scene::set_updated(bool updated)
{
    _updated = updated;
    for (auto& object : _object_array) {
        object->geometry()->set_updated(updated);
    }
}

int Scene::num_triangles()
{
    int num_triangles = 0;
    for (auto& object : _object_array) {
        num_triangles += object->geometry()->num_faces();
    }
    return num_triangles;
}
}
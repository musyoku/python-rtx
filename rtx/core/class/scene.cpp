#include "scene.h"

namespace rtx {
void Scene::add(std::shared_ptr<Object> object)
{
    _object_array.emplace_back(object);
    _updated = true;
}
bool Scene::updated()
{
    return _updated;
}
void Scene::set_updated(bool updated)
{
    _updated = updated;
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
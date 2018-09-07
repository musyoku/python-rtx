#include "scene.h"

namespace rtx {
void Scene::add(std::shared_ptr<Mesh> mesh)
{
    _mesh_array.emplace_back(mesh);
    _updated = true;
}
void Scene::add(std::shared_ptr<Light> light)
{
    _light_array.emplace_back(light);
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
    for (auto& mesh : _mesh_array) {
        num_triangles += mesh->_geometry->num_faces();
    }
    return num_triangles;
}
}
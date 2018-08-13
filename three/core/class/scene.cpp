#include "scene.h"

namespace three {
void Scene::add(std::shared_ptr<Mesh> mesh)
{
    _mesh_array.emplace_back(mesh);
}
}
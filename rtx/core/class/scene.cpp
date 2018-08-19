#include "scene.h"

namespace rtx {
void Scene::add(std::shared_ptr<Mesh> mesh)
{
    _mesh_array.emplace_back(mesh);
}
}
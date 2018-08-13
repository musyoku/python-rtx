#include "scene.h"

namespace three {
void Scene::add(std::shared_ptr<Mesh> mesh)
{
    _meshes.emplace_back(mesh);
}
}
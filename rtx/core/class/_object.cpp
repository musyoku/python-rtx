#include "light.h"
#include <glm/gtc/matrix_transform.hpp>

namespace rtx {
namespace py = pybind11;
bool Object::bvh_enabled() const
{
    return false;
}
int Object::bvh_max_triangles_per_node() const
{
    return -1;
}
}